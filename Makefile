# Real-time Voice Translation Pipeline Makefile

.PHONY: help build up down logs clean download-models bench test health

# Default target
help:
	@echo "Real-time Voice Translation Pipeline"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build all Docker images"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  logs           - Show logs for all services"
	@echo "  clean          - Clean up containers and volumes"
	@echo "  download-models - Download ML models (one-time setup)"
	@echo "  bench          - Run all benchmarks"
	@echo "  test           - Run health checks"
	@echo "  health         - Check service health"
	@echo ""
	@echo "Quick start:"
	@echo "  make download-models  # First time only"
	@echo "  make up              # Start all services"
	@echo "  open http://localhost:5173"

# Build all images
build:
	@echo "Building Docker images..."
	docker compose build

# Start all services
up:
	@echo "Starting all services..."
	docker compose up -d
	@echo "Services started! Access the app at http://localhost:5173"

# Stop all services
down:
	@echo "Stopping all services..."
	docker compose down

# Show logs
logs:
	@echo "Showing logs for all services..."
	docker compose logs -f

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker compose down -v
	docker system prune -f
	@echo "Cleanup complete!"

# Download models
download-models:
	@echo "Downloading ML models..."
	docker compose --profile download up model-downloader
	@echo "Models downloaded successfully!"

# Run all benchmarks
bench:
	@echo "Running all benchmarks..."
	@echo "1. Latency test..."
	python bench/latency_test.py --trials 3
	@echo "2. BLEU evaluation..."
	python bench/bleu_eval.py
	@echo "3. MOS evaluation..."
	python bench/mos_proxy.py
	@echo "All benchmarks completed!"

# Run health checks
test:
	@echo "Running health checks..."
	@echo "ASR Service:"
	@curl -s http://localhost:8000/health | jq . || echo "ASR service not responding"
	@echo "MT Service:"
	@curl -s http://localhost:8001/health | jq . || echo "MT service not responding"
	@echo "TTS Service:"
	@curl -s http://localhost:8002/health | jq . || echo "TTS service not responding"
	@echo "MLflow Service:"
	@curl -s http://localhost:5000/health || echo "MLflow service not responding"

# Check service health
health:
	@echo "Checking service health..."
	@echo "Frontend: http://localhost:5173"
	@echo "ASR: http://localhost:8000/health"
	@echo "MT: http://localhost:8001/health"
	@echo "TTS: http://localhost:8002/health"
	@echo "MLflow: http://localhost:5000"

# Development targets
dev-asr:
	@echo "Starting ASR service in development mode..."
	cd asr && python app.py

dev-mt:
	@echo "Starting MT service in development mode..."
	cd mt && python app.py

dev-tts:
	@echo "Starting TTS service in development mode..."
	cd tts && python app.py

dev-frontend:
	@echo "Starting frontend in development mode..."
	cd frontend && npm run dev

# Benchmark individual services
bench-latency:
	@echo "Running latency benchmark..."
	python bench/latency_test.py --trials 5

bench-bleu:
	@echo "Running BLEU evaluation..."
	python bench/bleu_eval.py

bench-mos:
	@echo "Running MOS evaluation..."
	python bench/mos_proxy.py

# GPU setup
setup-gpu:
	@echo "Setting up GPU support..."
	@echo "Make sure NVIDIA Container Toolkit is installed:"
	@echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
	@echo ""
	@echo "Test GPU access:"
	docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Performance monitoring
monitor:
	@echo "Monitoring system performance..."
	@echo "GPU usage:"
	@nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || echo "GPU not available"
	@echo ""
	@echo "Docker stats:"
	@docker stats --no-stream

# Quick start for new users
quickstart:
	@echo "Quick Start Guide"
	@echo "================"
	@echo "1. Copy environment file:"
	@echo "   cp env.example .env"
	@echo ""
	@echo "2. Download models (first time only):"
	@echo "   make download-models"
	@echo ""
	@echo "3. Start all services:"
	@echo "   make up"
	@echo ""
	@echo "4. Access the application:"
	@echo "   open http://localhost:5173"
	@echo ""
	@echo "5. Run benchmarks:"
	@echo "   make bench"
