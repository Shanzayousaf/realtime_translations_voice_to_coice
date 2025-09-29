"""
TTS Service - Streaming Text-to-Speech
Provides real-time audio synthesis using FastSpeech2 + HiFi-GAN or TTS fallback.
"""

import asyncio
import base64
import json
import logging
import os
import time
from typing import Optional, Dict, Any, List
import io
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import mlflow
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
TTS_HOST = os.getenv("TTS_HOST", "0.0.0.0")
TTS_PORT = int(os.getenv("TTS_PORT", "8002"))
TTS_MODEL = os.getenv("TTS_MODEL", "fastspeech2_small")
HIFIGAN_MODEL = os.getenv("HIFIGAN_MODEL", "hifigan_v1")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
TTS_RUNTIME = os.getenv("TTS_RUNTIME", "pt")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialize FastAPI app
app = FastAPI(title="TTS Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
tts_model = None
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
sequence_counter = 0
active_connections: Dict[str, WebSocket] = {}

# Pydantic models
class SynthesisRequest(BaseModel):
    text: str
    sequence: int
    language: str
    final: bool = True

class AudioChunk(BaseModel):
    type: str
    sequence: int
    pcm16_base64: str
    sample_rate: int
    timestamp: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    model_name: str

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_tts_model():
    """Load TTS model with optimizations for low latency"""
    global tts_model, device
    
    logger.info(f"Loading TTS model: {TTS_MODEL} on device: {device}")
    
    try:
        # Try to load TTS model (Coqui TTS as fallback)
        from TTS.api import TTS
        
        # Use a lightweight model for low latency
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"  # Fast and multilingual
        
        tts_model = TTS(model_name=model_name, progress_bar=False, gpu=USE_GPU)
        
        # Move to device if needed
        if hasattr(tts_model, 'to'):
            tts_model = tts_model.to(device)
        
        logger.info(f"TTS model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        logger.info("Falling back to espeak-ng")
        return False

def synthesize_speech_fallback(text: str, language: str = "en") -> np.ndarray:
    """Fallback TTS using espeak-ng"""
    try:
        import subprocess
        import tempfile
        
        # Create temporary file for audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Use espeak-ng to generate speech
        cmd = [
            "espeak-ng",
            "-v", language,
            "-s", "150",  # Speed
            "-w", tmp_path,
            text
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load audio file
        audio, sample_rate = torchaudio.load(tmp_path)
        audio = audio.numpy().flatten()
        
        # Clean up
        os.unlink(tmp_path)
        
        # Resample to 22050 Hz if needed
        if sample_rate != 22050:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
        
        return audio.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Fallback TTS failed: {e}")
        # Return silence
        return np.zeros(int(22050 * 0.5), dtype=np.float32)  # 0.5 seconds of silence

def synthesize_speech(text: str, language: str = "en") -> np.ndarray:
    """Synthesize speech with streaming optimizations"""
    try:
        if tts_model is not None:
            # Use TTS model
            audio = tts_model.tts(
                text=text,
                language=language,
                speaker_wav=None,  # Use default speaker
                split_sentences=False  # Don't split for streaming
            )
            
            # Convert to numpy array
            if isinstance(audio, list):
                audio = np.concatenate(audio)
            
            # Ensure correct format
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            return audio.astype(np.float32)
        
        else:
            # Use fallback
            return synthesize_speech_fallback(text, language)
            
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        return synthesize_speech_fallback(text, language)

def audio_to_pcm16(audio: np.ndarray, sample_rate: int = 22050) -> bytes:
    """Convert audio array to PCM16 bytes"""
    try:
        # Ensure audio is in range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        pcm16 = (audio * 32767).astype(np.int16)
        
        # Convert to bytes (little-endian)
        return pcm16.tobytes()
        
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return b""

def chunk_audio(audio: np.ndarray, chunk_duration_ms: int = 100) -> List[np.ndarray]:
    """Split audio into chunks for streaming"""
    try:
        sample_rate = 22050
        chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        
        chunks = []
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        return chunks
        
    except Exception as e:
        logger.error(f"Audio chunking failed: {e}")
        return [audio] if len(audio) > 0 else []

async def stream_audio_to_websocket(websocket: WebSocket, audio: np.ndarray, sequence: int):
    """Stream audio chunks to WebSocket"""
    try:
        # Split audio into chunks
        chunks = chunk_audio(audio, chunk_duration_ms=100)
        
        for i, chunk in enumerate(chunks):
            # Convert to PCM16
            pcm16_bytes = audio_to_pcm16(chunk)
            
            if len(pcm16_bytes) > 0:
                # Encode as base64
                pcm16_base64 = base64.b64encode(pcm16_bytes).decode('utf-8')
                
                # Create audio chunk message
                audio_chunk = AudioChunk(
                    type="audio_chunk",
                    sequence=sequence,
                    pcm16_base64=pcm16_base64,
                    sample_rate=22050,
                    timestamp=time.time()
                )
                
                # Send to WebSocket
                await websocket.send_text(audio_chunk.json())
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
        
        # Send final message
        final_message = {
            "type": "audio_final",
            "sequence": sequence,
            "duration": len(audio) / 22050,
            "timestamp": time.time()
        }
        
        await websocket.send_text(json.dumps(final_message))
        
    except Exception as e:
        logger.error(f"Audio streaming failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting TTS service...")
    
    # Load TTS model
    if not load_tts_model():
        logger.warning("TTS model loading failed, using fallback")
    
    logger.info("TTS service started successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=tts_model is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
        model_name=TTS_MODEL
    )

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "active_connections": len(active_connections),
        "device": device,
        "model": TTS_MODEL,
        "sequence_counter": sequence_counter
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, channelId: str = None):
    """WebSocket endpoint for streaming audio synthesis"""
    await websocket.accept()
    
    if channelId:
        active_connections[channelId] = websocket
        logger.info(f"Client connected: {channelId}")
    
    try:
        while True:
            # Receive synthesis request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Process synthesis request
            request = SynthesisRequest(**request_data)
            
            # Synthesize speech
            start_time = time.time()
            audio = synthesize_speech(request.text, request.language)
            synthesis_time = time.time() - start_time
            
            # Stream audio to client
            await stream_audio_to_websocket(websocket, audio, request.sequence)
            
            # Log metrics
            try:
                with mlflow.start_run():
                    mlflow.log_metric("tts_latency_ms", synthesis_time * 1000)
                    mlflow.log_metric("audio_duration_ms", len(audio) / 22050 * 1000)
                    mlflow.log_text(request.text, "synthesis_text.txt")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
            
            logger.info(f"Synthesized: '{request.text[:50]}...' ({synthesis_time*1000:.1f}ms)")
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {channelId}")
        if channelId and channelId in active_connections:
            del active_connections[channelId]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if channelId and channelId in active_connections:
            del active_connections[channelId]

@app.post("/synthesize")
async def synthesize_endpoint(request: SynthesisRequest):
    """REST endpoint for speech synthesis"""
    try:
        start_time = time.time()
        
        # Synthesize speech
        audio = synthesize_speech(request.text, request.language)
        
        # Convert to PCM16
        pcm16_bytes = audio_to_pcm16(audio)
        
        # Encode as base64
        pcm16_base64 = base64.b64encode(pcm16_bytes).decode('utf-8')
        
        synthesis_time = time.time() - start_time
        
        # Log metrics
        try:
            with mlflow.start_run():
                mlflow.log_metric("tts_latency_ms", synthesis_time * 1000)
                mlflow.log_metric("audio_duration_ms", len(audio) / 22050 * 1000)
                mlflow.log_text(request.text, "synthesis_text.txt")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
        
        return {
            "status": "success",
            "sequence": request.sequence,
            "audio_base64": pcm16_base64,
            "sample_rate": 22050,
            "duration": len(audio) / 22050,
            "synthesis_time_ms": synthesis_time * 1000
        }
        
    except Exception as e:
        logger.error(f"Synthesis endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_stream")
async def synthesize_stream_endpoint(request: SynthesisRequest):
    """Streaming endpoint for speech synthesis"""
    try:
        # Synthesize speech
        audio = synthesize_speech(request.text, request.language)
        
        # Split into chunks
        chunks = chunk_audio(audio, chunk_duration_ms=100)
        
        def generate_audio_chunks():
            for i, chunk in enumerate(chunks):
                pcm16_bytes = audio_to_pcm16(chunk)
                if len(pcm16_bytes) > 0:
                    yield pcm16_bytes
        
        return StreamingResponse(
            generate_audio_chunks(),
            media_type="audio/wav",
            headers={
                "Content-Type": "audio/wav",
                "X-Sequence": str(request.sequence),
                "X-Sample-Rate": "22050"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_available_voices():
    """Get available voices/languages"""
    return {
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "fr", "name": "French"},
            {"code": "es", "name": "Spanish"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"}
        ],
        "default_language": "en",
        "sample_rate": 22050
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=TTS_HOST,
        port=TTS_PORT,
        log_level="info",
        access_log=True
    )
