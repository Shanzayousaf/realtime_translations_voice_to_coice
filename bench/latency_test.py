"""
Latency Test Script
Measures end-to-end latency for the voice translation pipeline.
"""

import asyncio
import json
import time
import wave
import numpy as np
import websockets
import aiohttp
import logging
from typing import List, Dict, Any
import statistics
import mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ASR_WS_URL = "ws://localhost:8000/ws"
MT_URL = "http://localhost:8001/translate"
TTS_URL = "http://localhost:8002/synthesize"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Test utterances
TEST_UTTERANCES = [
    "Hello, how are you today?",
    "I would like to order a coffee please.",
    "What time is it?",
    "Can you help me with directions?",
    "Thank you very much for your assistance.",
    "I don't understand what you mean.",
    "Could you please repeat that?",
    "Where is the nearest restaurant?",
    "I need to make a phone call.",
    "What is the weather like today?"
]

def generate_test_audio(text: str, duration: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic audio for testing"""
    # Generate a simple sine wave as test audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, len(audio))
    audio = audio + noise
    
    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio_16bit = (audio * 32767).astype(np.int16)
    
    # Convert to bytes
    return audio_16bit.tobytes()

async def test_asr_latency(audio_data: bytes, utterance: str) -> Dict[str, Any]:
    """Test ASR latency"""
    start_time = time.time()
    
    try:
        async with websockets.connect(f"{ASR_WS_URL}?channelId=test&src=en&tgt=fr") as websocket:
            # Send audio data
            await websocket.send(audio_data)
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            asr_time = time.time() - start_time
            
            return {
                "asr_latency_ms": asr_time * 1000,
                "transcription": data.get("text", ""),
                "status": "success"
            }
            
    except Exception as e:
        logger.error(f"ASR test failed: {e}")
        return {
            "asr_latency_ms": -1,
            "transcription": "",
            "status": "failed",
            "error": str(e)
        }

async def test_mt_latency(text: str) -> Dict[str, Any]:
    """Test MT latency"""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                MT_URL,
                json={
                    "text": text,
                    "sequence": 1,
                    "language": "en"
                },
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as response:
                data = await response.json()
                mt_time = time.time() - start_time
                
                return {
                    "mt_latency_ms": mt_time * 1000,
                    "translation": data.get("text", ""),
                    "status": "success"
                }
                
    except Exception as e:
        logger.error(f"MT test failed: {e}")
        return {
            "mt_latency_ms": -1,
            "translation": "",
            "status": "failed",
            "error": str(e)
        }

async def test_tts_latency(text: str) -> Dict[str, Any]:
    """Test TTS latency"""
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TTS_URL,
                json={
                    "text": text,
                    "sequence": 1,
                    "language": "fr",
                    "final": True
                },
                timeout=aiohttp.ClientTimeout(total=15.0)
            ) as response:
                data = await response.json()
                tts_time = time.time() - start_time
                
                return {
                    "tts_latency_ms": tts_time * 1000,
                    "audio_duration_ms": data.get("duration", 0) * 1000,
                    "status": "success"
                }
                
    except Exception as e:
        logger.error(f"TTS test failed: {e}")
        return {
            "tts_latency_ms": -1,
            "audio_duration_ms": 0,
            "status": "failed",
            "error": str(e)
        }

async def test_end_to_end_latency(utterance: str) -> Dict[str, Any]:
    """Test complete end-to-end latency"""
    logger.info(f"Testing utterance: '{utterance}'")
    
    # Generate test audio
    audio_data = generate_test_audio(utterance)
    
    # Test ASR
    asr_result = await test_asr_latency(audio_data, utterance)
    if asr_result["status"] != "success":
        return asr_result
    
    # Test MT
    mt_result = await test_mt_latency(asr_result["transcription"])
    if mt_result["status"] != "success":
        return mt_result
    
    # Test TTS
    tts_result = await test_tts_latency(mt_result["translation"])
    if tts_result["status"] != "success":
        return tts_result
    
    # Calculate total latency
    total_latency = (
        asr_result["asr_latency_ms"] +
        mt_result["mt_latency_ms"] +
        tts_result["tts_latency_ms"]
    )
    
    return {
        "utterance": utterance,
        "asr_latency_ms": asr_result["asr_latency_ms"],
        "mt_latency_ms": mt_result["mt_latency_ms"],
        "tts_latency_ms": tts_result["tts_latency_ms"],
        "total_latency_ms": total_latency,
        "transcription": asr_result["transcription"],
        "translation": mt_result["translation"],
        "status": "success"
    }

async def run_latency_benchmark(num_trials: int = 10) -> List[Dict[str, Any]]:
    """Run latency benchmark"""
    logger.info(f"Starting latency benchmark with {num_trials} trials per utterance")
    
    results = []
    
    for utterance in TEST_UTTERANCES:
        utterance_results = []
        
        for trial in range(num_trials):
            logger.info(f"Trial {trial + 1}/{num_trials} for: '{utterance}'")
            
            result = await test_end_to_end_latency(utterance)
            utterance_results.append(result)
            
            # Small delay between trials
            await asyncio.sleep(1)
        
        # Calculate statistics for this utterance
        successful_results = [r for r in utterance_results if r["status"] == "success"]
        
        if successful_results:
            latencies = [r["total_latency_ms"] for r in successful_results]
            
            utterance_stats = {
                "utterance": utterance,
                "trials": len(successful_results),
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "success_rate": len(successful_results) / len(utterance_results)
            }
        else:
            utterance_stats = {
                "utterance": utterance,
                "trials": 0,
                "mean_latency_ms": -1,
                "median_latency_ms": -1,
                "std_latency_ms": -1,
                "min_latency_ms": -1,
                "max_latency_ms": -1,
                "p95_latency_ms": -1,
                "success_rate": 0
            }
        
        results.append(utterance_stats)
        logger.info(f"Utterance '{utterance}' - Mean latency: {utterance_stats['mean_latency_ms']:.1f}ms")
    
    return results

def log_to_mlflow(results: List[Dict[str, Any]]):
    """Log results to MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        with mlflow.start_run(run_name="latency_benchmark"):
            # Log overall statistics
            all_latencies = [r["mean_latency_ms"] for r in results if r["mean_latency_ms"] > 0]
            
            if all_latencies:
                mlflow.log_metric("overall_mean_latency_ms", statistics.mean(all_latencies))
                mlflow.log_metric("overall_median_latency_ms", statistics.median(all_latencies))
                mlflow.log_metric("overall_p95_latency_ms", np.percentile(all_latencies, 95))
                mlflow.log_metric("overall_min_latency_ms", min(all_latencies))
                mlflow.log_metric("overall_max_latency_ms", max(all_latencies))
                mlflow.log_metric("total_utterances", len(results))
                mlflow.log_metric("successful_utterances", len(all_latencies))
            
            # Log detailed results
            mlflow.log_text(json.dumps(results, indent=2), "latency_results.json")
            
            logger.info("Results logged to MLflow successfully")
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")

def print_summary(results: List[Dict[str, Any]]):
    """Print benchmark summary"""
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r["mean_latency_ms"] > 0]
    
    if not successful_results:
        print("❌ No successful tests completed")
        return
    
    all_latencies = [r["mean_latency_ms"] for r in successful_results]
    
    print(f"Total utterances tested: {len(results)}")
    print(f"Successful utterances: {len(successful_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    print()
    print("LATENCY STATISTICS:")
    print(f"  Mean latency: {statistics.mean(all_latencies):.1f} ms")
    print(f"  Median latency: {statistics.median(all_latencies):.1f} ms")
    print(f"  Std deviation: {statistics.stdev(all_latencies):.1f} ms")
    print(f"  Min latency: {min(all_latencies):.1f} ms")
    print(f"  Max latency: {max(all_latencies):.1f} ms")
    print(f"  95th percentile: {np.percentile(all_latencies, 95):.1f} ms")
    print()
    
    # Check if target latency is met
    target_latency = 700  # ms
    p95_latency = np.percentile(all_latencies, 95)
    
    if p95_latency <= target_latency:
        print(f"✅ TARGET LATENCY MET: P95 latency ({p95_latency:.1f}ms) <= {target_latency}ms")
    else:
        print(f"❌ TARGET LATENCY NOT MET: P95 latency ({p95_latency:.1f}ms) > {target_latency}ms")
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    for result in results:
        if result["mean_latency_ms"] > 0:
            print(f"'{result['utterance'][:50]}...'")
            print(f"  Mean: {result['mean_latency_ms']:.1f}ms, "
                  f"P95: {result['p95_latency_ms']:.1f}ms, "
                  f"Success: {result['success_rate']*100:.1f}%")
        else:
            print(f"'{result['utterance'][:50]}...' - FAILED")
    
    print("=" * 80)

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run latency benchmark")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per utterance")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    
    logger.info("Starting latency benchmark...")
    
    # Run benchmark
    results = await run_latency_benchmark(args.trials)
    
    # Print summary
    print_summary(results)
    
    # Log to MLflow
    if not args.no_mlflow:
        log_to_mlflow(results)
    
    logger.info("Latency benchmark completed")

if __name__ == "__main__":
    asyncio.run(main())
