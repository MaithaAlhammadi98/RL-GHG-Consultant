#!/bin/bash
# ============================================================================
# RL-Enhanced GHG Consultant - Docker Startup Script
# ============================================================================
# Easy script to start the application with Docker
# ============================================================================

set -e  # Exit on any error

echo "🚀 Starting RL-Enhanced GHG Consultant..."
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "📝 Please copy env.example to .env and add your API keys:"
    echo "   cp env.example .env"
    echo "   # Then edit .env with your API keys"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "🐳 Please start Docker Desktop and try again"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/baseline logs/qlearning logs/ppo logs/comparisons
mkdir -p src/data
mkdir -p chroma_persistent_storage

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting application..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:7860/ > /dev/null 2>&1; then
    echo "✅ Application is running!"
    echo "🌐 Open your browser and go to: http://localhost:7860"
    echo ""
    echo "📊 Available features:"
    echo "   • Three-way bot comparison (Baseline vs Q-Learning vs PPO)"
    echo "   • Interactive Q-Learning training with 👍👎 feedback"
    echo "   • Real-time Q-table updates"
    echo "   • Comprehensive GHG consulting responses"
    echo ""
    echo "🛑 To stop the application, run: docker-compose down"
    echo "📋 To view logs, run: docker-compose logs -f"
else
    echo "❌ Application failed to start!"
    echo "📋 Check logs with: docker-compose logs"
    exit 1
fi
