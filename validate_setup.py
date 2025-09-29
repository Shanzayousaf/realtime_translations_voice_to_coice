#!/usr/bin/env python3
"""
Setup Validation Script
Validates that all required files and configurations are present.
"""

import os
import sys
import yaml
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_docker_compose():
    """Validate docker-compose.yml syntax"""
    try:
        with open('docker-compose.yml', 'r') as f:
            yaml.safe_load(f)
        print("✅ docker-compose.yml: Valid YAML syntax")
        return True
    except Exception as e:
        print(f"❌ docker-compose.yml: Invalid syntax - {e}")
        return False

def check_environment_file():
    """Check if environment file exists"""
    if os.path.exists('.env'):
        print("✅ .env file: Present")
        return True
    elif os.path.exists('env.example'):
        print("⚠️  .env file: Missing, but env.example exists")
        print("   Run: cp env.example .env")
        return False
    else:
        print("❌ .env file: Missing")
        return False

def main():
    """Main validation function"""
    print("🔍 Validating Real-time Voice Translation Pipeline Setup")
    print("=" * 60)
    
    # Required files
    required_files = [
        ("docker-compose.yml", "Docker Compose configuration"),
        ("env.example", "Environment template"),
        ("README.md", "Project documentation"),
        ("Makefile", "Build automation"),
        ("asr/app.py", "ASR service"),
        ("asr/Dockerfile", "ASR container"),
        ("asr/requirements.txt", "ASR dependencies"),
        ("mt/app.py", "MT service"),
        ("mt/Dockerfile", "MT container"),
        ("mt/requirements.txt", "MT dependencies"),
        ("tts/app.py", "TTS service"),
        ("tts/Dockerfile", "TTS container"),
        ("tts/requirements.txt", "TTS dependencies"),
        ("frontend/package.json", "Frontend dependencies"),
        ("frontend/Dockerfile", "Frontend container"),
        ("frontend/src/App.jsx", "React application"),
        ("model_downloader/download_models.py", "Model downloader"),
        ("model_downloader/Dockerfile", "Model downloader container"),
        ("bench/latency_test.py", "Latency benchmark"),
        ("bench/bleu_eval.py", "BLEU evaluation"),
        ("bench/mos_proxy.py", "MOS evaluation"),
    ]
    
    # Check all required files
    all_files_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_present = False
    
    print("\n" + "=" * 60)
    
    # Check docker-compose syntax
    docker_compose_valid = check_docker_compose()
    
    # Check environment file
    env_file_ok = check_environment_file()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_files_present and docker_compose_valid:
        print("✅ All required files are present")
        print("✅ Docker Compose configuration is valid")
        
        if env_file_ok:
            print("✅ Environment configuration is ready")
            print("\n🚀 Setup is complete! You can now run:")
            print("   make download-models  # First time only")
            print("   make up              # Start all services")
            print("   open http://localhost:5173")
        else:
            print("⚠️  Environment file needs to be created")
            print("   Run: cp env.example .env")
        
        return 0
    else:
        print("❌ Setup validation failed")
        print("   Please check the missing files and fix any issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
