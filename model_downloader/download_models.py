"""
Model Downloader Service
Downloads all required models for the voice translation pipeline.
"""

import os
import json
import logging
import time
from typing import Dict, Any
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import whisper
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
MT_MODEL = os.getenv("MT_MODEL", "facebook/m2m100_418M")
TTS_MODEL = os.getenv("TTS_MODEL", "fastspeech2_small")
HIFIGAN_MODEL = os.getenv("HIFIGAN_MODEL", "hifigan_v1")

# Models directory
MODELS_DIR = "/app/models"

def create_models_directory():
    """Create models directory if it doesn't exist"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Created models directory: {MODELS_DIR}")

def download_whisper_model():
    """Download Whisper model"""
    logger.info(f"Downloading Whisper model: {WHISPER_MODEL}")
    
    try:
        start_time = time.time()
        
        # Download Whisper model
        model = whisper.load_model(
            WHISPER_MODEL,
            download_root=MODELS_DIR
        )
        
        download_time = time.time() - start_time
        logger.info(f"Whisper model downloaded successfully in {download_time:.2f}s")
        
        return {
            "model": "whisper",
            "version": WHISPER_MODEL,
            "size_mb": get_model_size_mb(MODELS_DIR),
            "download_time_s": download_time,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {e}")
        return {
            "model": "whisper",
            "version": WHISPER_MODEL,
            "status": "failed",
            "error": str(e)
        }

def download_translation_model():
    """Download translation model"""
    logger.info(f"Downloading translation model: {MT_MODEL}")
    
    try:
        start_time = time.time()
        
        # Download model and tokenizer
        if "m2m100" in MT_MODEL.lower():
            model = M2M100ForConditionalGeneration.from_pretrained(
                MT_MODEL,
                cache_dir=MODELS_DIR
            )
            tokenizer = M2M100Tokenizer.from_pretrained(
                MT_MODEL,
                cache_dir=MODELS_DIR
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MT_MODEL,
                cache_dir=MODELS_DIR
            )
            tokenizer = AutoTokenizer.from_pretrained(
                MT_MODEL,
                cache_dir=MODELS_DIR
            )
        
        download_time = time.time() - start_time
        logger.info(f"Translation model downloaded successfully in {download_time:.2f}s")
        
        return {
            "model": "translation",
            "version": MT_MODEL,
            "size_mb": get_model_size_mb(MODELS_DIR),
            "download_time_s": download_time,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to download translation model: {e}")
        return {
            "model": "translation",
            "version": MT_MODEL,
            "status": "failed",
            "error": str(e)
        }

def download_tts_model():
    """Download TTS model"""
    logger.info(f"Downloading TTS model: {TTS_MODEL}")
    
    try:
        start_time = time.time()
        
        # Download TTS model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        
        download_time = time.time() - start_time
        logger.info(f"TTS model downloaded successfully in {download_time:.2f}s")
        
        return {
            "model": "tts",
            "version": TTS_MODEL,
            "size_mb": get_model_size_mb(MODELS_DIR),
            "download_time_s": download_time,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to download TTS model: {e}")
        return {
            "model": "tts",
            "version": TTS_MODEL,
            "status": "failed",
            "error": str(e)
        }

def get_model_size_mb(directory: str) -> float:
    """Calculate total size of models directory in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def save_manifest(results: Dict[str, Any]):
    """Save download manifest"""
    manifest = {
        "timestamp": time.time(),
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": results,
        "total_size_mb": sum(r.get("size_mb", 0) for r in results.values()),
        "total_download_time_s": sum(r.get("download_time_s", 0) for r in results.values())
    }
    
    manifest_path = os.path.join(MODELS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to: {manifest_path}")

def main():
    """Main download function"""
    logger.info("Starting model download process...")
    
    # Create models directory
    create_models_directory()
    
    # Download models
    results = {}
    
    # Download Whisper
    results["whisper"] = download_whisper_model()
    
    # Download Translation model
    results["translation"] = download_translation_model()
    
    # Download TTS model
    results["tts"] = download_tts_model()
    
    # Save manifest
    save_manifest(results)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)
    
    total_size = 0
    total_time = 0
    successful = 0
    
    for model_name, result in results.items():
        status = result.get("status", "unknown")
        size_mb = result.get("size_mb", 0)
        download_time = result.get("download_time_s", 0)
        
        logger.info(f"{model_name.upper()}: {status}")
        if status == "success":
            logger.info(f"  Size: {size_mb:.1f} MB")
            logger.info(f"  Time: {download_time:.1f} s")
            total_size += size_mb
            total_time += download_time
            successful += 1
        else:
            logger.info(f"  Error: {result.get('error', 'Unknown error')}")
    
    logger.info("=" * 50)
    logger.info(f"Total models downloaded: {successful}/{len(results)}")
    logger.info(f"Total size: {total_size:.1f} MB")
    logger.info(f"Total time: {total_time:.1f} s")
    logger.info("=" * 50)
    
    if successful == len(results):
        logger.info("✅ All models downloaded successfully!")
        return 0
    else:
        logger.error("❌ Some models failed to download")
        return 1

if __name__ == "__main__":
    exit(main())
