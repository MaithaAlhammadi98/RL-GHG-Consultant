@echo off
REM ============================================================================
REM RL-Enhanced GHG Consultant - Docker Startup Script (Windows)
REM ============================================================================
REM Easy script to start the application with Docker on Windows
REM ============================================================================

echo 🚀 Starting RL-Enhanced GHG Consultant...
echo ==========================================

REM Check if .env file exists (in parent directory)
if not exist ..\.env (
    echo ❌ .env file not found!
    echo 📝 Please copy env.example to .env and add your API keys:
    echo    copy env.example .env
    echo    # Then edit .env with your API keys
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running!
    echo 🐳 Please start Docker Desktop and try again
    pause
    exit /b 1
)

REM Create necessary directories (in parent directory)
echo 📁 Creating directories...
if not exist ..\logs mkdir ..\logs
if not exist ..\logs\baseline mkdir ..\logs\baseline
if not exist ..\logs\qlearning mkdir ..\logs\qlearning
if not exist ..\logs\ppo mkdir ..\logs\ppo
if not exist ..\logs\comparisons mkdir ..\logs\comparisons
if not exist ..\src\data mkdir ..\src\data
if not exist ..\chroma_persistent_storage mkdir ..\chroma_persistent_storage

REM Build and start the application
echo 🔨 Building Docker image...
docker-compose build

echo 🚀 Starting application...
docker-compose up -d

REM Wait for the application to start
echo ⏳ Waiting for application to start...
timeout /t 10 /nobreak >nul

REM Check if the application is running
curl -f http://localhost:7860/ >nul 2>&1
if errorlevel 1 (
    echo ❌ Application failed to start!
    echo 📋 Check logs with: docker-compose logs
    pause
    exit /b 1
) else (
    echo ✅ Application is running!
    echo 🌐 Open your browser and go to: http://localhost:7860
    echo.
    echo 📊 Available features:
    echo    • Three-way bot comparison (Baseline vs Q-Learning vs PPO)
    echo    • Interactive Q-Learning training with 👍👎 feedback
    echo    • Real-time Q-table updates
    echo    • Comprehensive GHG consulting responses
    echo.
    echo 🛑 To stop the application, run: docker-compose down
    echo 📋 To view logs, run: docker-compose logs -f
    echo.
    echo Press any key to open the application in your browser...
    pause >nul
    start http://localhost:7860
)
