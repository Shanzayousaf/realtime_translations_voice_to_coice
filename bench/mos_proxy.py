"""
MOS (Mean Opinion Score) Proxy Script
Evaluates audio quality using MOS prediction or manual evaluation.
"""

import asyncio
import json
import logging
import time
import base64
import numpy as np
from typing import List, Dict, Any
import aiohttp
import mlflow
from scipy.io import wavfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TTS_URL = "http://localhost:8002/synthesize"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Test texts for MOS evaluation
TEST_TEXTS = [
    "Hello, how are you today?",
    "I would like to order a coffee please.",
    "What time is it?",
    "Can you help me with directions?",
    "Thank you very much for your assistance.",
    "I don't understand what you mean.",
    "Could you please repeat that?",
    "Where is the nearest restaurant?",
    "I need to make a phone call.",
    "What is the weather like today?",
    "Good morning, have a nice day!",
    "Excuse me, where is the bathroom?",
    "I'm sorry, I don't speak French.",
    "How much does this cost?",
    "I would like to check in to my hotel."
]

class MOSPredictor:
    """Simple MOS predictor based on audio quality metrics"""
    
    def __init__(self):
        self.sample_rate = 22050
    
    def calculate_audio_metrics(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate audio quality metrics"""
        try:
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Calculate SNR (simplified)
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data - np.mean(audio_data))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Calculate spectral centroid
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            spectral_centroid = np.sum(np.abs(fft) * freqs) / np.sum(np.abs(fft))
            
            # Calculate zero crossing rate
            zcr = np.mean(np.abs(np.diff(np.sign(audio_data))))
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Calculate dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio_data)) / (np.mean(np.abs(audio_data)) + 1e-10))
            
            return {
                "snr_db": snr,
                "spectral_centroid_hz": spectral_centroid,
                "zero_crossing_rate": zcr,
                "rms_energy": rms,
                "dynamic_range_db": dynamic_range,
                "duration_s": len(audio_data) / self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate audio metrics: {e}")
            return {
                "snr_db": 0.0,
                "spectral_centroid_hz": 0.0,
                "zero_crossing_rate": 0.0,
                "rms_energy": 0.0,
                "dynamic_range_db": 0.0,
                "duration_s": 0.0
            }
    
    def predict_mos(self, audio_data: np.ndarray) -> float:
        """Predict MOS score based on audio quality metrics"""
        metrics = self.calculate_audio_metrics(audio_data)
        
        # Simple MOS prediction based on heuristics
        # This is a simplified model - in practice, you'd use a trained MOS predictor
        
        # Base score
        mos_score = 3.0
        
        # SNR contribution (higher SNR = better quality)
        snr_score = min(2.0, max(0.0, metrics["snr_db"] / 20.0))
        mos_score += snr_score
        
        # Dynamic range contribution
        dr_score = min(1.0, max(0.0, metrics["dynamic_range_db"] / 40.0))
        mos_score += dr_score
        
        # RMS energy contribution (moderate energy is good)
        rms_score = 1.0 - abs(metrics["rms_energy"] - 0.1) * 5.0
        rms_score = max(0.0, min(1.0, rms_score))
        mos_score += rms_score
        
        # Zero crossing rate contribution (moderate ZCR is good)
        zcr_score = 1.0 - abs(metrics["zero_crossing_rate"] - 0.1) * 5.0
        zcr_score = max(0.0, min(1.0, zcr_score))
        mos_score += zcr_score
        
        # Clamp to valid MOS range (1-5)
        mos_score = max(1.0, min(5.0, mos_score))
        
        return mos_score

async def synthesize_audio(text: str, language: str = "fr") -> np.ndarray:
    """Synthesize audio using TTS service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TTS_URL,
                json={
                    "text": text,
                    "sequence": 1,
                    "language": language,
                    "final": True
                },
                timeout=aiohttp.ClientTimeout(total=15.0)
            ) as response:
                data = await response.json()
                
                if "audio_base64" in data:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(data["audio_base64"])
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
                    
                    return audio_data
                else:
                    logger.error("No audio data in response")
                    return np.array([])
                    
    except Exception as e:
        logger.error(f"Audio synthesis failed: {e}")
        return np.array([])

async def evaluate_audio_quality() -> List[Dict[str, Any]]:
    """Evaluate audio quality using MOS prediction"""
    logger.info("Starting MOS evaluation...")
    
    mos_predictor = MOSPredictor()
    results = []
    
    for i, text in enumerate(TEST_TEXTS):
        logger.info(f"Evaluating {i+1}/{len(TEST_TEXTS)}: '{text[:50]}...'")
        
        # Synthesize audio
        audio_data = await synthesize_audio(text)
        
        if len(audio_data) > 0:
            # Calculate audio metrics
            metrics = mos_predictor.calculate_audio_metrics(audio_data)
            
            # Predict MOS score
            predicted_mos = mos_predictor.predict_mos(audio_data)
            
            result = {
                "text": text,
                "predicted_mos": predicted_mos,
                "audio_metrics": metrics,
                "audio_duration_s": len(audio_data) / 22050,
                "status": "success"
            }
            
            logger.info(f"  Predicted MOS: {predicted_mos:.2f}")
            logger.info(f"  Duration: {result['audio_duration_s']:.2f}s")
            
        else:
            result = {
                "text": text,
                "predicted_mos": 0.0,
                "audio_metrics": {},
                "audio_duration_s": 0.0,
                "status": "failed"
            }
            logger.warning(f"  Failed to synthesize audio")
        
        results.append(result)
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    return results

def calculate_mos_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate MOS statistics"""
    successful_results = [r for r in results if r["status"] == "success"]
    
    if not successful_results:
        return {
            "mean_mos": 0.0,
            "median_mos": 0.0,
            "std_mos": 0.0,
            "min_mos": 0.0,
            "max_mos": 0.0,
            "num_samples": 0,
            "success_rate": 0.0
        }
    
    mos_scores = [r["predicted_mos"] for r in successful_results]
    
    return {
        "mean_mos": np.mean(mos_scores),
        "median_mos": np.median(mos_scores),
        "std_mos": np.std(mos_scores),
        "min_mos": np.min(mos_scores),
        "max_mos": np.max(mos_scores),
        "num_samples": len(successful_results),
        "success_rate": len(successful_results) / len(results)
    }

def log_to_mlflow(results: List[Dict[str, Any]], stats: Dict[str, float]):
    """Log results to MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        with mlflow.start_run(run_name="mos_evaluation"):
            # Log MOS statistics
            mlflow.log_metric("mean_mos", stats["mean_mos"])
            mlflow.log_metric("median_mos", stats["median_mos"])
            mlflow.log_metric("std_mos", stats["std_mos"])
            mlflow.log_metric("min_mos", stats["min_mos"])
            mlflow.log_metric("max_mos", stats["max_mos"])
            mlflow.log_metric("num_samples", stats["num_samples"])
            mlflow.log_metric("success_rate", stats["success_rate"])
            
            # Log detailed results
            mlflow.log_text(json.dumps(results, indent=2), "mos_results.json")
            
            logger.info("Results logged to MLflow successfully")
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")

def print_summary(results: List[Dict[str, Any]], stats: Dict[str, float]):
    """Print evaluation summary"""
    print("\n" + "=" * 80)
    print("MOS EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"Total samples evaluated: {len(results)}")
    print(f"Successful samples: {stats['num_samples']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print()
    
    if stats['num_samples'] > 0:
        print("MOS SCORES:")
        print(f"  Mean MOS: {stats['mean_mos']:.2f}")
        print(f"  Median MOS: {stats['median_mos']:.2f}")
        print(f"  Std Deviation: {stats['std_mos']:.2f}")
        print(f"  Min MOS: {stats['min_mos']:.2f}")
        print(f"  Max MOS: {stats['max_mos']:.2f}")
        print()
        
        # Quality assessment
        mean_mos = stats['mean_mos']
        if mean_mos >= 4.0:
            print("✅ EXCELLENT AUDIO QUALITY")
        elif mean_mos >= 3.0:
            print("✅ GOOD AUDIO QUALITY")
        elif mean_mos >= 2.0:
            print("⚠️  FAIR AUDIO QUALITY")
        else:
            print("❌ POOR AUDIO QUALITY")
        
        print("\nSAMPLE EVALUATIONS:")
        print("-" * 80)
        for i, result in enumerate(results[:5]):  # Show first 5 examples
            if result["status"] == "success":
                print(f"\nExample {i+1}:")
                print(f"  Text: {result['text']}")
                print(f"  Predicted MOS: {result['predicted_mos']:.2f}")
                print(f"  Duration: {result['audio_duration_s']:.2f}s")
                if result['audio_metrics']:
                    metrics = result['audio_metrics']
                    print(f"  SNR: {metrics.get('snr_db', 0):.1f} dB")
                    print(f"  Dynamic Range: {metrics.get('dynamic_range_db', 0):.1f} dB")
        
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more examples")
    else:
        print("❌ No successful audio synthesis to evaluate")
    
    print("\nMANUAL EVALUATION INSTRUCTIONS:")
    print("-" * 80)
    print("For manual MOS evaluation:")
    print("1. Listen to each synthesized audio sample")
    print("2. Rate quality on a scale of 1-5:")
    print("   5 = Excellent (natural, clear, no artifacts)")
    print("   4 = Good (mostly natural, minor artifacts)")
    print("   3 = Fair (understandable, some artifacts)")
    print("   2 = Poor (difficult to understand, many artifacts)")
    print("   1 = Bad (unintelligible, severe artifacts)")
    print("3. Record scores and calculate mean MOS")
    
    print("=" * 80)

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MOS evaluation")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    
    logger.info("Starting MOS evaluation...")
    
    # Run evaluation
    results = await evaluate_audio_quality()
    
    # Calculate statistics
    stats = calculate_mos_statistics(results)
    
    # Print summary
    print_summary(results, stats)
    
    # Log to MLflow
    if not args.no_mlflow:
        log_to_mlflow(results, stats)
    
    logger.info("MOS evaluation completed")

if __name__ == "__main__":
    asyncio.run(main())
