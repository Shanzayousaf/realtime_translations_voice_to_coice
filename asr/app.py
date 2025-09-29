"""
ASR Service - Streaming Whisper-based Speech Recognition
Provides WebSocket endpoint for real-time audio transcription with low latency.
"""

import asyncio
import json
import logging
import os
import time
import wave
from typing import Optional, Dict, Any
import numpy as np
import torch
import whisper
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import mlflow
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
ASR_HOST = os.getenv("ASR_HOST", "0.0.0.0")
ASR_PORT = int(os.getenv("ASR_PORT", "8000"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
VAD_ENABLED = os.getenv("VAD_ENABLED", "true").lower() == "true"
CHUNK_DURATION_MS = int(os.getenv("CHUNK_DURATION_MS", "200"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "16"))
MT_SERVICE_URL = os.getenv("MT_SERVICE_URL", "http://mt:8001")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialize FastAPI app
app = FastAPI(title="ASR Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
vad = None
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
sequence_counter = 0
active_connections: Dict[str, WebSocket] = {}
mt_queue = asyncio.Queue(maxsize=MAX_QUEUE)

# Pydantic models
class TranscriptionRequest(BaseModel):
    text: str
    final: bool
    sequence: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_whisper_model():
    """Load Whisper model with optimizations for low latency"""
    global model, device
    
    logger.info(f"Loading Whisper model: {WHISPER_MODEL} on device: {device}")
    
    try:
        # Load model with optimizations
        model = whisper.load_model(
            WHISPER_MODEL, 
            device=device,
            download_root="/app/models"
        )
        
        # Optimize for inference
        if device == "cuda":
            model.half()  # Use fp16 for faster inference
            torch.backends.cudnn.benchmark = True
            
        # Compile model for PyTorch 2.0+ if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        logger.info(f"Whisper model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return False

def initialize_vad():
    """Initialize Voice Activity Detection"""
    global vad
    
    if VAD_ENABLED:
        try:
            # VAD for 16kHz audio, frame duration in ms
            vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
            logger.info("VAD initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            vad = None

def preprocess_audio(audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Preprocess raw audio data for Whisper"""
    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1]
        audio_array = audio_array / 32768.0
        
        # Resample if necessary (Whisper expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        return audio_array
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return None

def detect_voice_activity(audio_data: bytes) -> bool:
    """Detect voice activity in audio chunk"""
    if not VAD_ENABLED or vad is None:
        return True  # Skip VAD if disabled
    
    try:
        # VAD expects specific frame sizes
        frame_duration_ms = 20  # 20ms frames
        frame_size = int(16000 * frame_duration_ms / 1000)  # 320 samples
        
        # Check if audio is long enough
        if len(audio_data) < frame_size * 2:
            return True  # Process short chunks anyway
        
        # Check voice activity
        return vad.is_speech(audio_data, 16000)
        
    except Exception as e:
        logger.warning(f"VAD check failed: {e}")
        return True  # Process if VAD fails

async def transcribe_audio(audio_data: bytes, language: str = "en") -> Dict[str, Any]:
    """Transcribe audio using Whisper with streaming optimizations"""
    global sequence_counter
    
    try:
        # Preprocess audio
        audio_array = preprocess_audio(audio_data)
        if audio_array is None:
            return None
        
        # Check voice activity
        if not detect_voice_activity(audio_data):
            return None
        
        # Transcribe with streaming-optimized parameters
        with torch.no_grad():
            result = model.transcribe(
                audio_array,
                language=language,
                task="transcribe",
                fp16=device == "cuda",
                # Streaming optimizations
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,  # Faster for streaming
                initial_prompt=None,  # No context for lower latency
                word_timestamps=False,  # Disable for speed
                verbose=False
            )
        
        sequence_counter += 1
        
        return {
            "text": result["text"].strip(),
            "sequence": sequence_counter,
            "language": language,
            "confidence": getattr(result, 'no_speech_prob', 0.0)
        }
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

async def send_to_mt(transcription_data: Dict[str, Any]):
    """Send transcription to MT service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MT_SERVICE_URL}/translate",
                json=transcription_data,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Translation sent to MT: {result.get('text', '')[:50]}...")
                else:
                    logger.error(f"MT service error: {response.status}")
                    
    except Exception as e:
        logger.error(f"Failed to send to MT service: {e}")

async def process_audio_chunk(websocket: WebSocket, audio_data: bytes, language: str):
    """Process audio chunk and send results"""
    try:
        # Transcribe audio
        transcription = await transcribe_audio(audio_data, language)
        
        if transcription and transcription["text"]:
            # Send partial transcription to frontend
            partial_message = {
                "type": "asr_partial",
                "sequence": transcription["sequence"],
                "text": transcription["text"],
                "final": False,
                "timestamp": time.time(),
                "confidence": transcription["confidence"]
            }
            
            await websocket.send_text(json.dumps(partial_message))
            
            # Send to MT service
            await send_to_mt(transcription)
            
            # Log to MLflow
            try:
                with mlflow.start_run():
                    mlflow.log_metric("asr_latency_ms", time.time() * 1000)
                    mlflow.log_text(transcription["text"], "transcription.txt")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting ASR service...")
    
    # Load Whisper model
    if not load_whisper_model():
        raise RuntimeError("Failed to load Whisper model")
    
    # Initialize VAD
    initialize_vad()
    
    logger.info("ASR service started successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device,
        gpu_available=torch.cuda.is_available()
    )

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "active_connections": len(active_connections),
        "queue_size": mt_queue.qsize(),
        "device": device,
        "model": WHISPER_MODEL
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, channelId: str = None, src: str = "en", tgt: str = "fr"):
    """WebSocket endpoint for streaming audio"""
    await websocket.accept()
    
    if channelId:
        active_connections[channelId] = websocket
        logger.info(f"Client connected: {channelId}")
    
    try:
        while True:
            # Receive audio data as binary
            audio_data = await websocket.receive_bytes()
            
            # Process audio in background to avoid blocking
            asyncio.create_task(
                process_audio_chunk(websocket, audio_data, src)
            )
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {channelId}")
        if channelId and channelId in active_connections:
            del active_connections[channelId]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if channelId and channelId in active_connections:
            del active_connections[channelId]

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """REST endpoint for single transcription"""
    try:
        # This is a simplified version for testing
        # In production, you'd want to handle audio data properly
        return {
            "text": request.text,
            "sequence": request.sequence,
            "final": request.final
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=ASR_HOST,
        port=ASR_PORT,
        log_level="info",
        access_log=True
    )
