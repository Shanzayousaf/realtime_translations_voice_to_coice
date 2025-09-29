"""
BLEU Evaluation Script
Computes BLEU scores for translation quality evaluation.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Tuple
import aiohttp
import mlflow
from sacrebleu import BLEU, CHRF, TER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ASR_WS_URL = "ws://localhost:8000/ws"
MT_URL = "http://localhost:8001/translate"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Test dataset (source, reference translation)
TEST_DATASET = [
    ("Hello, how are you today?", "Bonjour, comment allez-vous aujourd'hui ?"),
    ("I would like to order a coffee please.", "Je voudrais commander un café s'il vous plaît."),
    ("What time is it?", "Quelle heure est-il ?"),
    ("Can you help me with directions?", "Pouvez-vous m'aider avec les directions ?"),
    ("Thank you very much for your assistance.", "Merci beaucoup pour votre aide."),
    ("I don't understand what you mean.", "Je ne comprends pas ce que vous voulez dire."),
    ("Could you please repeat that?", "Pourriez-vous répéter cela s'il vous plaît ?"),
    ("Where is the nearest restaurant?", "Où est le restaurant le plus proche ?"),
    ("I need to make a phone call.", "J'ai besoin de passer un appel téléphonique."),
    ("What is the weather like today?", "Quel temps fait-il aujourd'hui ?"),
    ("Good morning, have a nice day!", "Bonjour, passez une bonne journée !"),
    ("Excuse me, where is the bathroom?", "Excusez-moi, où sont les toilettes ?"),
    ("I'm sorry, I don't speak French.", "Je suis désolé, je ne parle pas français."),
    ("How much does this cost?", "Combien cela coûte-t-il ?"),
    ("I would like to check in to my hotel.", "Je voudrais m'enregistrer à mon hôtel."),
    ("Is there a pharmacy nearby?", "Y a-t-il une pharmacie à proximité ?"),
    ("I'm lost, can you help me?", "Je suis perdu, pouvez-vous m'aider ?"),
    ("What is your name?", "Quel est votre nom ?"),
    ("Nice to meet you!", "Ravi de vous rencontrer !"),
    ("Have a good evening!", "Passez une bonne soirée !")
]

async def test_asr_transcription(text: str) -> str:
    """Test ASR transcription"""
    try:
        import websockets
        import numpy as np
        
        # Generate synthetic audio (simple sine wave)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        audio = (audio * 32767).astype(np.int16).tobytes()
        
        async with websockets.connect(f"{ASR_WS_URL}?channelId=bleu_test&src=en&tgt=fr") as websocket:
            await websocket.send(audio)
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            return data.get("text", "")
            
    except Exception as e:
        logger.error(f"ASR test failed: {e}")
        return ""

async def test_translation(text: str) -> str:
    """Test translation"""
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
                return data.get("text", "")
                
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        return ""

async def evaluate_translation_quality() -> List[Dict[str, Any]]:
    """Evaluate translation quality using BLEU scores"""
    logger.info("Starting BLEU evaluation...")
    
    results = []
    all_predictions = []
    all_references = []
    
    for i, (source, reference) in enumerate(TEST_DATASET):
        logger.info(f"Evaluating {i+1}/{len(TEST_DATASET)}: '{source[:50]}...'")
        
        # Test ASR (simplified - just return source text for now)
        # In real evaluation, you'd use actual audio
        asr_text = source  # source  # await test_asr_transcription(source)
        
        # Test translation
        translation = await test_translation(asr_text)
        
        if translation:
            all_predictions.append(translation)
            all_references.append(reference)
            
            result = {
                "source": source,
                "reference": reference,
                "prediction": translation,
                "asr_text": asr_text
            }
            results.append(result)
            
            logger.info(f"  Source: {source}")
            logger.info(f"  Reference: {reference}")
            logger.info(f"  Prediction: {translation}")
        else:
            logger.warning(f"  Failed to translate: {source}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    return results, all_predictions, all_references

def calculate_bleu_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores"""
    try:
        # Calculate BLEU score
        bleu = BLEU()
        bleu_score = bleu.corpus_score(predictions, [references])
        
        # Calculate CHRF score
        chrf = CHRF()
        chrf_score = chrf.corpus_score(predictions, [references])
        
        # Calculate TER score
        ter = TER()
        ter_score = ter.corpus_score(predictions, [references])
        
        return {
            "bleu_score": bleu_score.score,
            "bleu_brevity_penalty": bleu_score.bp,
            "bleu_ratios": bleu_score.ratios,
            "chrf_score": chrf_score.score,
            "ter_score": ter_score.score,
            "num_samples": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate BLEU scores: {e}")
        return {
            "bleu_score": 0.0,
            "bleu_brevity_penalty": 0.0,
            "bleu_ratios": [0.0, 0.0, 0.0, 0.0],
            "chrf_score": 0.0,
            "ter_score": 0.0,
            "num_samples": 0
        }

def log_to_mlflow(results: List[Dict[str, Any]], scores: Dict[str, float]):
    """Log results to MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        with mlflow.start_run(run_name="bleu_evaluation"):
            # Log BLEU scores
            mlflow.log_metric("bleu_score", scores["bleu_score"])
            mlflow.log_metric("bleu_brevity_penalty", scores["bleu_brevity_penalty"])
            mlflow.log_metric("chrf_score", scores["chrf_score"])
            mlflow.log_metric("ter_score", scores["ter_score"])
            mlflow.log_metric("num_samples", scores["num_samples"])
            
            # Log individual BLEU ratios
            for i, ratio in enumerate(scores["bleu_ratios"]):
                mlflow.log_metric(f"bleu_ratio_{i+1}", ratio)
            
            # Log detailed results
            mlflow.log_text(json.dumps(results, indent=2, ensure_ascii=False), "bleu_results.json")
            
            logger.info("Results logged to MLflow successfully")
            
    except Exception as e:
        logger.error(f"Failed to log to MLflow: {e}")

def print_summary(results: List[Dict[str, Any]], scores: Dict[str, float]):
    """Print evaluation summary"""
    print("\n" + "=" * 80)
    print("BLEU EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"Total samples evaluated: {scores['num_samples']}")
    print()
    print("TRANSLATION QUALITY SCORES:")
    print(f"  BLEU Score: {scores['bleu_score']:.4f}")
    print(f"  BLEU Brevity Penalty: {scores['bleu_brevity_penalty']:.4f}")
    print(f"  CHRF Score: {scores['chrf_score']:.4f}")
    print(f"  TER Score: {scores['ter_score']:.4f}")
    print()
    
    # BLEU ratio breakdown
    print("BLEU N-gram Ratios:")
    for i, ratio in enumerate(scores['bleu_ratios']):
        print(f"  {i+1}-gram: {ratio:.4f}")
    print()
    
    # Quality assessment
    bleu_score = scores['bleu_score']
    if bleu_score >= 0.3:
        print("✅ GOOD TRANSLATION QUALITY")
    elif bleu_score >= 0.2:
        print("⚠️  MODERATE TRANSLATION QUALITY")
    else:
        print("❌ POOR TRANSLATION QUALITY")
    
    print("\nSAMPLE TRANSLATIONS:")
    print("-" * 80)
    for i, result in enumerate(results[:5]):  # Show first 5 examples
        print(f"\nExample {i+1}:")
        print(f"  Source: {result['source']}")
        print(f"  Reference: {result['reference']}")
        print(f"  Prediction: {result['prediction']}")
    
    if len(results) > 5:
        print(f"\n... and {len(results) - 5} more examples")
    
    print("=" * 80)

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BLEU evaluation")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    
    logger.info("Starting BLEU evaluation...")
    
    # Run evaluation
    results, predictions, references = await evaluate_translation_quality()
    
    if not predictions:
        logger.error("No successful translations to evaluate")
        return
    
    # Calculate scores
    scores = calculate_bleu_scores(predictions, references)
    
    # Print summary
    print_summary(results, scores)
    
    # Log to MLflow
    if not args.no_mlflow:
        log_to_mlflow(results, scores)
    
    logger.info("BLEU evaluation completed")

if __name__ == "__main__":
    asyncio.run(main())
