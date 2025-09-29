# Real-time Voice-to-Voice Translation Pipeline

A complete, containerized real-time voice translation system with sub-700ms latency target. This pipeline captures microphone audio, performs streaming speech recognition, machine translation, and text-to-speech synthesis, delivering translated audio in near real-time.

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd realtime_translations_voice_to_coice

# Copy environment configuration
cp env.example .env

# Download models (one-time setup)
docker compose --profile download up model-downloader

# Start all services
docker compose up --build

# Access the application
open http://localhost:5173
```

## 🏗️ Architecture

The system consists of 6 microservices:

- **ASR Service** (`asr/`): Streaming speech recognition using Whisper
- **MT Service** (`mt/`): Real-time machine translation using M2M100
- **TTS Service** (`tts/`): Audio synthesis using TTS models
- **Frontend** (`frontend/`): React web interface with WebRTC audio capture
- **Model Downloader** (`model_downloader/`): Downloads and caches ML models
- **MLflow** (`mlflow/`): Experiment tracking and metrics logging

## 📋 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8-16GB VRAM (recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 20GB+ free space for models

### Software Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)

## ⚙️ Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
# Model Configuration
WHISPER_MODEL=small                    # tiny, base, small, medium, large
MT_MODEL=facebook/m2m100_418M         # Translation model
TTS_MODEL=fastspeech2_small           # TTS model
HIFIGAN_MODEL=hifigan_v1              # Vocoder model

# Service Configuration
ASR_HOST=0.0.0.0
ASR_PORT=8000
MT_HOST=0.0.0.0
MT_PORT=8001
TTS_HOST=0.0.0.0
TTS_PORT=8002

# GPU Configuration
USE_GPU=true
CUDA_VISIBLE_DEVICES=0

# Performance Tuning
VAD_ENABLED=true                      # Voice Activity Detection
CHUNK_DURATION_MS=200                 # Audio chunk size
MAX_QUEUE=16                          # Max queue size
TTS_RUNTIME=pt                        # pt, onnx, trt
```

### Model Selection for Latency Optimization

| Model | Size | Latency | Quality | Use Case |
|-------|------|---------|---------|----------|
| `whisper-tiny` | 39MB | ~100ms | Low | Fastest |
| `whisper-small` | 244MB | ~200ms | Good | **Recommended** |
| `whisper-medium` | 769MB | ~400ms | High | Quality priority |
| `whisper-large` | 1550MB | ~800ms | Best | Best quality |

## 🚀 Usage

### Starting the System

1. **Download Models** (first time only):
   ```bash
   docker compose --profile download up model-downloader
   ```

2. **Start All Services**:
   ```bash
   docker compose up --build
   ```

3. **Access the Web Interface**:
   - Open http://localhost:5173
   - Allow microphone permissions
   - Select source and target languages
   - Click "Start Recording" and speak

### Service Endpoints

- **Frontend**: http://localhost:5173
- **ASR WebSocket**: ws://localhost:8000/ws
- **MT API**: http://localhost:8001/translate
- **TTS WebSocket**: ws://localhost:8002/ws
- **MLflow UI**: http://localhost:5000

### WebSocket API

#### ASR WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws?channelId=test&src=en&tgt=fr');
ws.send(audioData); // PCM16 audio bytes
```

#### TTS WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8002/ws?channelId=test');
ws.send(JSON.stringify({
  text: "Hello world",
  sequence: 1,
  language: "fr",
  final: true
}));
```

## 📊 Benchmarking

### Latency Testing
```bash
# Run latency benchmark
python bench/latency_test.py --trials 5

# Skip MLflow logging
python bench/latency_test.py --no-mlflow
```

### Translation Quality (BLEU)
```bash
# Run BLEU evaluation
python bench/bleu_eval.py

# Skip MLflow logging
python bench/bleu_eval.py --no-mlflow
```

### Audio Quality (MOS)
```bash
# Run MOS evaluation
python bench/mos_proxy.py

# Skip MLflow logging
python bench/mos_proxy.py --no-mlflow
```

### All Benchmarks
```bash
# Run all benchmarks
make bench
```

## 🎯 Performance Optimization

### Achieving <700ms Latency

1. **Model Selection**:
   - Use `whisper-small` or `whisper-tiny`
   - Use `facebook/m2m100_418M` for translation
   - Use lightweight TTS models

2. **GPU Optimization**:
   ```bash
   # Enable GPU support
   USE_GPU=true
   CUDA_VISIBLE_DEVICES=0
   ```

3. **Audio Processing**:
   - Use smaller chunk sizes (100-200ms)
   - Enable VAD to skip silence
   - Use 16kHz sample rate

4. **Model Optimizations**:
   - Enable fp16 precision
   - Use torch.compile() (PyTorch 2.0+)
   - Export to ONNX for faster inference

### Advanced Optimizations

1. **ONNX Export** (for production):
   ```python
   # Export models to ONNX
   python scripts/export_to_onnx.py
   ```

2. **TensorRT Optimization** (NVIDIA):
   ```python
   # Convert to TensorRT
   python scripts/convert_to_tensorrt.py
   ```

3. **Model Quantization**:
   ```python
   # Quantize models for faster inference
   python scripts/quantize_models.py
   ```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Not Found**:
   ```bash
   # Check NVIDIA Container Toolkit
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   
   # Install NVIDIA Container Toolkit
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :5173
   
   # Kill processes using ports
   sudo fuser -k 5173/tcp
   ```

3. **Model Download Failures**:
   ```bash
   # Clear model cache
   docker volume rm realtime_translations_voice_to_coice_models
   
   # Re-download models
   docker compose --profile download up model-downloader
   ```

4. **CORS Issues**:
   - Ensure all services are running
   - Check service URLs in frontend configuration
   - Verify WebSocket connections

5. **High Latency**:
   - Check GPU utilization: `nvidia-smi`
   - Reduce model sizes in `.env`
   - Enable VAD and reduce chunk size
   - Check network latency between services

### Debug Mode

Enable debug logging:
```bash
# Set log level to debug
export LOG_LEVEL=DEBUG
docker compose up --build
```

### Health Checks

Check service health:
```bash
# Check all services
curl http://localhost:8000/health  # ASR
curl http://localhost:8001/health  # MT
curl http://localhost:8002/health  # TTS
curl http://localhost:5000/health  # MLflow
```

## 📈 Monitoring

### MLflow Dashboard
- Access: http://localhost:5000
- View experiments, metrics, and artifacts
- Compare different model configurations

### Service Metrics
- **ASR**: Latency, transcription accuracy, VAD efficiency
- **MT**: Translation latency, BLEU scores
- **TTS**: Synthesis latency, audio quality (MOS)

### Performance Metrics
- End-to-end latency (target: <700ms)
- Memory usage per service
- GPU utilization
- Audio quality scores

## 🛠️ Development

### Project Structure
```
├── asr/                    # ASR service
│   ├── app.py             # FastAPI WebSocket server
│   ├── Dockerfile         # Container definition
│   └── requirements.txt   # Python dependencies
├── mt/                    # MT service
│   ├── app.py             # Translation API
│   ├── Dockerfile
│   └── requirements.txt
├── tts/                   # TTS service
│   ├── app.py             # Audio synthesis
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/              # React web app
│   ├── src/               # Source code
│   ├── Dockerfile
│   └── package.json
├── model_downloader/      # Model management
│   ├── download_models.py
│   ├── Dockerfile
│   └── requirements.txt
├── bench/                 # Benchmarking scripts
│   ├── latency_test.py    # Latency measurement
│   ├── bleu_eval.py       # Translation quality
│   └── mos_proxy.py       # Audio quality
├── docker-compose.yml     # Service orchestration
├── env.example           # Environment template
└── README.md             # This file
```

### Adding New Models

1. **Update model configuration** in `env.example`
2. **Modify download script** in `model_downloader/download_models.py`
3. **Update service code** to use new model
4. **Test with benchmarks** to ensure performance

### Customizing Services

Each service can be customized by modifying:
- Model parameters in service code
- Environment variables
- Docker configurations
- API endpoints and message formats

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and benchmarks
5. Submit a pull request

## 📚 References

- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [M2M100 Paper](https://arxiv.org/abs/2010.11125)
- [FastSpeech2 Paper](https://arxiv.org/abs/2006.04558)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs: `docker compose logs <service>`
3. Open an issue on GitHub
4. Check MLflow for performance metrics

---

**Note**: This system is designed for demonstration and development purposes. For production use, consider additional security, scalability, and reliability measures.
