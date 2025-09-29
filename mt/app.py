"""
MT Service - Streaming Machine Translation
Provides real-time translation using M2M100 or NLLB models with low latency.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any, List
import torch
from transformers import (
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import mlflow
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MT_HOST = os.getenv("MT_HOST", "0.0.0.0")
MT_PORT = int(os.getenv("MT_PORT", "8001"))
MT_MODEL = os.getenv("MT_MODEL", "facebook/m2m100_418M")
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts:8002")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialize FastAPI app
app = FastAPI(title="MT Service", version="1.0.0")

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
tokenizer = None
device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
sequence_counter = 0

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    sequence: int
    language: str
    confidence: Optional[float] = None

class TranslationResponse(BaseModel):
    text: str
    sequence: int
    final: bool
    timestamp: float
    source_lang: str
    target_lang: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    model_name: str

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_translation_model():
    """Load translation model with optimizations for low latency"""
    global model, tokenizer, device
    
    logger.info(f"Loading translation model: {MT_MODEL} on device: {device}")
    
    try:
        # Load model and tokenizer
        if "m2m100" in MT_MODEL.lower():
            model = M2M100ForConditionalGeneration.from_pretrained(
                MT_MODEL,
                cache_dir="/app/models"
            )
            tokenizer = M2M100Tokenizer.from_pretrained(
                MT_MODEL,
                cache_dir="/app/models"
            )
        else:
            # Fallback to generic model
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MT_MODEL,
                cache_dir="/app/models"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MT_MODEL,
                cache_dir="/app/models"
            )
        
        # Move to device
        model = model.to(device)
        
        # Optimize for inference
        if device == "cuda":
            model.half()  # Use fp16 for faster inference
            torch.backends.cudnn.benchmark = True
        
        # Set to evaluation mode
        model.eval()
        
        # Compile model for PyTorch 2.0+ if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
        logger.info(f"Translation model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load translation model: {e}")
        return False

def detect_language(text: str) -> str:
    """Simple language detection (can be enhanced with langdetect)"""
    # Basic heuristic - in production, use proper language detection
    if any(char in text for char in "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"):
        return "fr"  # French
    elif any(char in text for char in "ñáéíóúü"):
        return "es"  # Spanish
    elif any(char in text for char in "äöüß"):
        return "de"  # German
    else:
        return "en"  # Default to English

def translate_text(text: str, source_lang: str = "en", target_lang: str = "fr") -> str:
    """Translate text with streaming optimizations"""
    global sequence_counter
    
    try:
        if not text.strip():
            return ""
        
        # Set source language for M2M100
        if hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = source_lang
        
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Limit for low latency
        ).to(device)
        
        # Generate translation with streaming optimizations
        with torch.no_grad():
            # Use greedy decoding for lowest latency
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(128, len(text.split()) * 2),  # Limit output length
                num_beams=1,  # Greedy decoding for speed
                do_sample=False,  # Deterministic for consistency
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
                no_repeat_ngram_size=2,  # Prevent n-gram repetition
                temperature=1.0,  # No sampling
                top_p=1.0,  # No nucleus sampling
                top_k=1,  # No top-k sampling
                use_cache=True  # Use KV cache for speed
            )
        
        # Decode output
        if hasattr(tokenizer, 'tgt_lang'):
            tokenizer.tgt_lang = target_lang
        
        translated_text = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        sequence_counter += 1
        
        return translated_text.strip()
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original text on error

async def send_to_tts(translation_data: Dict[str, Any]):
    """Send translation to TTS service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{TTS_SERVICE_URL}/synthesize",
                json=translation_data,
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Translation sent to TTS: {result.get('status', 'unknown')}")
                else:
                    logger.error(f"TTS service error: {response.status}")
                    
    except Exception as e:
        logger.error(f"Failed to send to TTS service: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting MT service...")
    
    # Load translation model
    if not load_translation_model():
        raise RuntimeError("Failed to load translation model")
    
    logger.info("MT service started successfully")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
        model_name=MT_MODEL
    )

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "device": device,
        "model": MT_MODEL,
        "sequence_counter": sequence_counter
    }

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    """Translate text with streaming support"""
    try:
        start_time = time.time()
        
        # Detect source language if not provided
        source_lang = request.language or detect_language(request.text)
        target_lang = "fr"  # Default target language
        
        # Translate text
        translated_text = translate_text(
            request.text, 
            source_lang=source_lang, 
            target_lang=target_lang
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = TranslationResponse(
            text=translated_text,
            sequence=request.sequence,
            final=True,  # For now, always final
            timestamp=time.time(),
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Send to TTS service
        tts_data = {
            "text": translated_text,
            "sequence": request.sequence,
            "language": target_lang,
            "final": True
        }
        asyncio.create_task(send_to_tts(tts_data))
        
        # Log to MLflow
        try:
            with mlflow.start_run():
                mlflow.log_metric("mt_latency_ms", latency_ms)
                mlflow.log_text(request.text, "source_text.txt")
                mlflow.log_text(translated_text, "translated_text.txt")
                mlflow.log_param("source_lang", source_lang)
                mlflow.log_param("target_lang", target_lang)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
        
        logger.info(f"Translated: '{request.text[:50]}...' -> '{translated_text[:50]}...' ({latency_ms:.1f}ms)")
        
        return response
        
    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate_batch")
async def translate_batch_endpoint(requests: List[TranslationRequest]):
    """Batch translation endpoint for efficiency"""
    try:
        start_time = time.time()
        results = []
        
        for request in requests:
            source_lang = request.language or detect_language(request.text)
            translated_text = translate_text(
                request.text, 
                source_lang=source_lang, 
                target_lang="fr"
            )
            
            results.append(TranslationResponse(
                text=translated_text,
                sequence=request.sequence,
                final=True,
                timestamp=time.time(),
                source_lang=source_lang,
                target_lang="fr"
            ))
        
        batch_latency_ms = (time.time() - start_time) * 1000
        
        # Log batch metrics
        try:
            with mlflow.start_run():
                mlflow.log_metric("batch_latency_ms", batch_latency_ms)
                mlflow.log_metric("batch_size", len(requests))
                mlflow.log_metric("avg_latency_per_item_ms", batch_latency_ms / len(requests))
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_supported_languages():
    """Get supported language pairs"""
    return {
        "supported_languages": [
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
        "default_source": "en",
        "default_target": "fr"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=MT_HOST,
        port=MT_PORT,
        log_level="info",
        access_log=True
    )
