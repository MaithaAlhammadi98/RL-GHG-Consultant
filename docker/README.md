# 🐳 Docker Setup for RL-Enhanced GHG Consultant

This folder contains all Docker-related files for easy deployment.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- API keys for OpenAI and Groq

### Setup
1. **Copy environment template:**
   ```bash
   cp env.example ../.env
   ```

2. **Edit `.env` file** with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

### Run Application

**Windows:**
```cmd
start_docker.bat
```

**Mac/Linux:**
```bash
./start_docker.sh
```

**Manual:**
```bash
docker-compose up -d
```

### Access
Open browser: `http://localhost:7860`

## 📁 Files in This Folder

- `Dockerfile` - Multi-stage Docker build
- `docker-compose.yml` - Deployment configuration
- `start_docker.sh` - Linux/Mac startup script
- `start_docker.bat` - Windows startup script
- `test_docker.py` - Setup verification script
- `README_DOCKER.md` - Detailed technical documentation
- `TEACHER_SETUP_GUIDE.md` - Simple guide for teachers

## 🛠️ Management

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose build --no-cache
```

## 📖 Documentation

- **Teachers**: See `TEACHER_SETUP_GUIDE.md`
- **Developers**: See `README_DOCKER.md`
- **Troubleshooting**: Run `python test_docker.py`
