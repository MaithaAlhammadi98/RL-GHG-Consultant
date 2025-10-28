# 🎓 Teacher Setup Guide - RL-Enhanced GHG Consultant

This guide will help you set up the RL experiment so you can run it anytime without worrying about expiring Gradio links.

## 🎯 What This Is

This is a **Reinforcement Learning experiment** that compares three different approaches to answering GHG (Greenhouse Gas) consulting questions:

1. **Baseline Bot**: Basic responses, no learning
2. **Q-Learning Bot**: Learns from feedback, shows Q-table updates
3. **PPO Bot**: Advanced neural network learning, best performance

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop/
- Install and start Docker Desktop
- Wait for it to show "Docker Desktop is running"

### Step 2: Get API Keys (Free)
You need two free API keys:

**OpenAI API Key:**
1. Go to: https://platform.openai.com/api-keys
2. Sign up (free account works)
3. Create a new API key
4. Copy it

**Groq API Key:**
1. Go to: https://console.groq.com/keys
2. Sign up (free account works)
3. Create a new API key
4. Copy it

### Step 3: Configure Environment
1. Copy `env.example` to `.env`
2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

### Step 4: Start the Application

**On Windows:**
- Double-click `start_docker.bat`

**On Mac/Linux:**
- Open terminal in the project folder
- Run: `./start_docker.sh`

**Manual way:**
- Open terminal in the project folder
- Run: `docker-compose up -d`

### Step 5: Access the Application
- Open your browser
- Go to: `http://localhost:7860`
- The application will load (first time may take 2-3 minutes)

## 🎮 How to Use the Demo

### Main Interface
- **Left Column**: Baseline Bot (basic responses)
- **Middle Column**: Q-Learning Bot (learns from feedback)
- **Right Column**: PPO Bot (best performance)

### Interactive Features
1. **Ask Questions**: Type any GHG-related question
2. **Compare Responses**: See how different methods answer
3. **Train Q-Learning**: Click 👍 or 👎 on Q-Learning responses
4. **Watch Learning**: See Q-table values update in real-time
5. **Sample Questions**: Click on pre-loaded examples

### Educational Value
- **Reinforcement Learning**: See how agents learn from feedback
- **RAG Systems**: Understand retrieval-augmented generation
- **Performance Comparison**: Compare different AI approaches
- **Interactive Learning**: Hands-on experience with RL

## 📊 What Students Will See

### Three Different Approaches
1. **Baseline**: Short, basic answers (150 words)
2. **Q-Learning**: Detailed answers with learned retrieval strategy
3. **PPO**: Most comprehensive answers with advanced learning

### Real-time Learning
- Q-Learning bot shows its "brain" (Q-table)
- Students can train it by giving feedback
- Watch Q-values change in real-time
- Understand how RL agents learn

### Performance Metrics
- Built-in comparison table
- Success rates and improvements
- Visual progress tracking
- Detailed experiment logs

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

### Restart Application
```bash
docker-compose down
docker-compose up -d
```

### Reset Everything
```bash
docker-compose down
docker system prune -f
docker-compose up -d
```

## 🔧 Troubleshooting

### "Docker is not running"
- Start Docker Desktop
- Wait for it to show "running" status
- Try again

### "Port 7860 already in use"
- Another application is using port 7860
- Stop other applications or change port in `docker-compose.yml`

### "Application won't start"
- Check API keys in `.env` file
- Run: `docker-compose logs` to see errors
- Try: `docker-compose down && docker-compose up -d`

### "Database is empty"
- First run needs to download models
- Wait 2-3 minutes for initial setup
- Check logs for progress

### "Slow performance"
- First run downloads models (slow)
- Subsequent runs are faster
- Close other applications to free RAM

## 📁 Data Persistence

All data is saved locally and persists between runs:
- **Q-Table**: `src/data/q_table.json`
- **Database**: `chroma_persistent_storage/`
- **Logs**: `logs/` folder
- **Results**: `logs/comparisons/`

## 🎓 Teaching Tips

### For Lectures
1. **Start with Baseline**: Show basic AI responses
2. **Introduce Q-Learning**: Explain the learning process
3. **Demonstrate PPO**: Show advanced neural learning
4. **Interactive Session**: Let students give feedback
5. **Compare Results**: Analyze performance differences

### For Assignments
1. **Experiment Design**: Have students design their own questions
2. **Feedback Analysis**: Study how different feedback affects learning
3. **Performance Comparison**: Analyze which method works best
4. **Code Review**: Examine the RL algorithms
5. **Extension Projects**: Modify the system

### Key Learning Objectives
- **Reinforcement Learning**: How agents learn from feedback
- **RAG Systems**: Combining retrieval with generation
- **Performance Evaluation**: Measuring AI system quality
- **Interactive Learning**: Real-time model training
- **Algorithm Comparison**: Different RL approaches

## 🔒 Security & Privacy

- **Local Only**: Everything runs on your computer
- **No Data Sharing**: All data stays local
- **API Keys**: Stored securely in `.env` file
- **No Internet**: Required only for initial setup

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Run: `python test_docker.py` for diagnostics
3. Check logs: `docker-compose logs`
4. Try restarting: `docker-compose down && docker-compose up -d`

## 🎉 Success!

Once everything is working, you'll have:
- ✅ A fully functional RL experiment
- ✅ Interactive learning demonstrations
- ✅ Persistent data and results
- ✅ No expiring links or external dependencies
- ✅ Educational value for students

**Happy Teaching! 🎓**

---

*This setup ensures your RL experiment runs reliably anytime, anywhere, without worrying about external services or expiring links.*
