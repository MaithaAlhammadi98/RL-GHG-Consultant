# 🐳 RL-Enhanced GHG Consultant - Docker Setup

This Docker setup allows your teacher to run the RL experiment anytime without worrying about expiring Gradio links. The application runs locally in a container and is accessible via `http://localhost:7860`.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- API keys for OpenAI and Groq

### 1. Setup Environment
```bash
# Copy the environment template
cp env.example .env

# Edit .env with your API keys
# You need:
# - OPENAI_API_KEY (for GPT-4o-mini judge)
# - GROQ_API_KEY (for Llama-3.1-8b-instant)
```

### 2. Start the Application

**On Windows:**
```cmd
start_docker.bat
```

**On Mac/Linux:**
```bash
chmod +x start_docker.sh
./start_docker.sh
```

**Manual Docker Compose:**
```bash
docker-compose up -d
```

### 3. Access the Application
Open your browser and go to: `http://localhost:7860`

## 🎯 Features Available

- **Three-Way Bot Comparison**: Baseline vs Q-Learning vs PPO
- **Interactive Learning**: Click 👍👎 to train Q-Learning bot live
- **Real-time Q-Table**: Watch Q-values update in real-time
- **Persistent Data**: All data persists between restarts
- **Comprehensive Logging**: All experiments saved to logs/

## 📊 What Your Teacher Will See

1. **Main Interface**: Three columns showing different bot responses
2. **Sample Questions**: Pre-loaded GHG consulting questions
3. **Interactive Training**: Feedback buttons to train Q-Learning
4. **Live Q-Table**: Real-time display of Q-values
5. **Experiment Results**: Built-in performance comparison

## 🛠️ Management Commands

### Start Application
```bash
docker-compose up -d
```

### Stop Application
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

### Rebuild Application
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Reset All Data
```bash
docker-compose down
docker volume prune -f
docker-compose up -d
```

## 📁 Data Persistence

The following data persists between restarts:
- **Q-Table**: `src/data/q_table.json`
- **Chroma Database**: `chroma_persistent_storage/`
- **Experiment Logs**: `logs/`
- **Generated Visualizations**: `logs/comparisons/`

## 🔧 Troubleshooting

### Application Won't Start
1. Check if Docker Desktop is running
2. Verify API keys in `.env` file
3. Check logs: `docker-compose logs`

### Port Already in Use
If port 7860 is busy, edit `docker-compose.yml`:
```yaml
ports:
  - "7861:7860"  # Use different port
```

### Database Issues
If the database is empty:
1. The app will show an error message
2. Run the database population script
3. Or restart the container

### Performance Issues
- The app uses CPU-only inference (no GPU)
- First run may be slow due to model downloads
- Subsequent runs are faster

## 📋 System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Docker**: Version 20.10+

## 🎓 Educational Value

This Docker setup demonstrates:
- **Reinforcement Learning**: Q-Learning and PPO algorithms
- **RAG Systems**: Retrieval-Augmented Generation
- **Interactive Learning**: Real-time model training
- **Performance Comparison**: Baseline vs RL methods
- **Containerization**: Modern deployment practices

## 🔒 Security Notes

- API keys are stored in `.env` file (not committed to git)
- Application runs locally (no external exposure)
- All data stays on the local machine
- No internet required after initial setup

## 📞 Support

If you encounter issues:
1. Check the logs: `docker-compose logs`
2. Verify Docker is running: `docker info`
3. Ensure API keys are correct in `.env`
4. Try rebuilding: `docker-compose build --no-cache`

---

**Happy Learning! 🎉**

Your teacher can now run this experiment anytime without worrying about expiring links or complex setup procedures.
