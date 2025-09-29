"""
ONNX Export Script
Exports PyTorch models to ONNX format for optimized inference.
"""

import os
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_whisper_to_onnx(model_name="small", output_dir="models/onnx"):
    """Export Whisper model to ONNX"""
    try:
        import whisper
        import onnx
        from onnxruntime.tools import optimizer
        
        logger.info(f"Exporting Whisper {model_name} to ONNX...")
        
        # Load model
        model = whisper.load_model(model_name)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 80, 3000)  # mel spectrogram input
        
        # Export to ONNX
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"whisper_{model_name}.onnx")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mel'],
            output_names=['logits'],
            dynamic_axes={
                'mel': {2: 'sequence_length'},
                'logits': {2: 'sequence_length'}
            }
        )
        
        logger.info(f"Whisper ONNX model saved to: {onnx_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export Whisper to ONNX: {e}")
        return False

def export_translation_to_onnx(model_name="facebook/m2m100_418M", output_dir="models/onnx"):
    """Export translation model to ONNX"""
    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        
        logger.info(f"Exporting {model_name} to ONNX...")
        
        # Load model and tokenizer
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        
        model.eval()
        
        # Create dummy input
        dummy_text = "Hello world"
        inputs = tokenizer(dummy_text, return_tensors="pt")
        
        # Export to ONNX
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "m2m100.onnx")
        
        torch.onnx.export(
            model,
            (inputs.input_ids, inputs.attention_mask),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {1: 'sequence_length'},
                'attention_mask': {1: 'sequence_length'},
                'logits': {1: 'sequence_length'}
            }
        )
        
        logger.info(f"Translation ONNX model saved to: {onnx_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export translation model to ONNX: {e}")
        return False

def main():
    """Main export function"""
    logger.info("Starting ONNX export process...")
    
    # Export Whisper
    whisper_success = export_whisper_to_onnx()
    
    # Export Translation model
    mt_success = export_translation_to_onnx()
    
    # Summary
    logger.info("=" * 50)
    logger.info("ONNX EXPORT SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Whisper ONNX: {'✅ Success' if whisper_success else '❌ Failed'}")
    logger.info(f"Translation ONNX: {'✅ Success' if mt_success else '❌ Failed'}")
    
    if whisper_success and mt_success:
        logger.info("✅ All models exported successfully!")
        return 0
    else:
        logger.error("❌ Some models failed to export")
        return 1

if __name__ == "__main__":
    exit(main())
