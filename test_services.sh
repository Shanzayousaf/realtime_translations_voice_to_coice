#!/bin/bash
# Test script for voice translation services

echo "🧪 Testing Voice Translation Services"
echo "===================================="

# Function to test HTTP endpoint
test_endpoint() {
    local url=$1
    local name=$2
    
    echo -n "Testing $name... "
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "✅ OK"
    else
        echo "❌ FAILED"
    fi
}

# Test all service health endpoints
echo "Health Checks:"
test_endpoint "http://localhost:8000/health" "ASR Service"
test_endpoint "http://localhost:8001/health" "MT Service" 
test_endpoint "http://localhost:8002/health" "TTS Service"
test_endpoint "http://localhost:5000/health" "MLflow Service"
test_endpoint "http://localhost:5173" "Frontend"

echo ""
echo "Service URLs:"
echo "Frontend: http://localhost:5173"
echo "ASR API: http://localhost:8000"
echo "MT API: http://localhost:8001" 
echo "TTS API: http://localhost:8002"
echo "MLflow: http://localhost:5000"

echo ""
echo "WebSocket URLs:"
echo "ASR WebSocket: ws://localhost:8000/ws"
echo "TTS WebSocket: ws://localhost:8002/ws"
