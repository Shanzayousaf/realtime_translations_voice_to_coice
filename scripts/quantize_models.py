"""
Model Quantization Script
Quantizes models for faster inference with minimal quality loss.
"""

import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_whisper_model(model_name="small", output_dir="models/quantized"):
    """Quantize Whisper model"""
    try:
        import whisper
        
        logger.info(f"Quantizing Whisper {model_name}...")
        
        # Load model
        model = whisper.load_model(model_name)
        
        # Quantize to int8
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv1d}, 
            dtype=torch.qint8
        )
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"whisper_{model_name}_quantized.pt")
        torch.save(quantized_model.state_dict(), model_path)
        
        logger.info(f"Quantized Whisper model saved to: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to quantize Whisper model: {e}")
        return False

def quantize_translation_model(model_name="facebook/m2m100_418M", output_dir="models/quantized"):
    """Quantize translation model"""
    try:
        from transformers import M2M100ForConditionalGeneration
        
        logger.info(f"Quantizing {model_name}...")
        
        # Load model
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
        # Quantize to int8
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Embedding},
            dtype=torch.qint8
        )
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "m2m100_quantized.pt")
        torch.save(quantized_model.state_dict(), model_path)
        
        logger.info(f"Quantized translation model saved to: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to quantize translation model: {e}")
        return False

def main():
    """Main quantization function"""
    logger.info("Starting model quantization process...")
    
    # Quantize Whisper
    whisper_success = quantize_whisper_model()
    
    # Quantize Translation model
    mt_success = quantize_translation_model()
    
    # Summary
    logger.info("=" * 50)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Whisper Quantized: {'✅ Success' if whisper_success else '❌ Failed'}")
    logger.info(f"Translation Quantized: {'✅ Success' if mt_success else '❌ Failed'}")
    
    if whisper_success and mt_success:
        logger.info("✅ All models quantized successfully!")
        return 0
    else:
        logger.error("❌ Some models failed to quantize")
        return 1

if __name__ == "__main__":
    exit(main())
