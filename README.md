# 🌍 RL-Enhanced GHG Consultant Chatbot

> **Fine-Tuning Language Models through Reinforcement Learning for GHG Compliance**  
> *By The Rewards Musketeers*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tested on Windows & macOS](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey)]()

An intelligent GHG (Greenhouse Gas) consulting chatbot that uses **Reinforcement Learning** to optimize document retrieval in RAG (Retrieval-Augmented Generation) systems. We demonstrate that RL agents (Q-Learning & PPO) learn superior retrieval policies compared to fixed baseline strategies.

---

## 🚀 Overview

Traditional RAG systems use static retrieval strategies. We improve this by training RL agents to **dynamically select optimal document filters** based on question characteristics. Our system learns which documents (legal, financial, technical) to retrieve for each query type, improving answer quality by **6-8%**.

**Key Innovation:** Multi-component reward function evaluating answer quality, retrieval relevance, grounding, and policy diversity — enabling interpretable, continuously improving chatbot behavior.

---

## 🧠 Architecture

```
User Question → State Encoder → RL Agent (Q-Learning/PPO) → Document Filter
                                           ↓
                                    RAG Process ↔ ChromaDB
                                           ↓
                                      LLM (Groq) → Answer
                                           ↓
                                   Reward Calculation ← Judge Evaluation
                                           ↓
                                   Update RL Agent
```

---

## 🎯 Results

*(N=40 test questions, evaluated by GPT-4o-mini judge)*

| Method | Avg Judge Score | Improvement | User Feedback |
|--------|-----------------|-------------|---------------|
| **Baseline** (no RL) | 0.830 | - | 100% 👍 |
| **Q-Learning** | 0.880 | +6.0% | 90% 👍 |
| **PPO** | 0.900 | +8.4% | 100% 👍 |

![Results Comparison](docs/images/complete_comparison_3methods.png)

→ **[Full Experiment Logs & Analysis](logs/comparisons/)**

---

## 🧩 Components

Our system consists of:

- **🔍 RAG Pipeline** – Retrieves relevant GHG documents using ChromaDB vector database
- **🎮 Q-Learning Agent** – Tabular RL learning retrieval policies (α=0.3, γ=0.9, ε=0.2)
- **🧠 PPO Agent** – Neural network-based policy optimization (clip=0.2, GAE λ=0.95)
- **🎯 Multi-Component Reward** – Evaluates quality, relevance, grounding, diversity (50%+20%+15%+15%)
- **⚖️ LLM-as-Judge** – GPT-4o-mini evaluates answers without ground truth access

### Reward Function Summary:
```python
total_reward = 0.5 * judge_score      # Answer quality [0,1]
             + 0.2 * retrieval_score  # Chunk relevance [0,1]
             + 0.15 * action_score    # Policy diversity [0,1]
             + 0.15 * grounding_score # Citation quality [0,1]
```

→ **[Complete Technical Documentation](docs/STUDY.md)** (2,350+ lines)

---

## ⚙️ Quick Start

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/MaithaAlhammadi98/RL-GHG-Consultant.git
cd RL-GHG-Consultant

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

### **2. Setup Environment**

```bash
# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY and OPENAI_API_KEY
```

### **3. Populate Database**

```bash
# Generate ChromaDB vector store from PDF documents
python populate_database.py
```

### **4. Run Interactive Demo**

```bash
# Launch Gradio interface at http://localhost:7860
python three_bot_demo.py
```

**Features:**
- 🤖 Compare all 3 bots side-by-side
- 👍👎 Train Q-Learning agent with live feedback
- 📊 Watch Q-table update in real-time

### **5. Run Full Experiment**

```bash
# Train & evaluate all methods, generate comparison plots
python complete_experiment.py
```

---

## 📁 Project Structure

```
RL-GHG-Consultant/
├── src/backend/           # Core RL & RAG implementation
│   ├── rl_agent.py       # Q-Learning agent
│   ├── ppo_agent.py      # PPO agent
│   ├── rag_process.py    # RAG pipeline
│   ├── reward_enhanced.py # Multi-component reward
│   └── state.py          # State encoder
├── three_bot_demo.py     # 🎮 Interactive Gradio demo
├── complete_experiment.py # 📊 Full experiment runner
├── populate_database.py  # 🗄️ Database setup
├── docs/                 # 📚 Documentation
│   ├── STUDY.md         # Complete technical guide
│   └── images/          # Result visualizations
├── logs/                 # 📈 Experiment results
│   ├── baseline/
│   ├── qlearning/
│   ├── ppo/
│   └── comparisons/
├── requirements.txt
├── Dockerfile
└── REPORT.md            # 📄 Final project report (TBD)
```

## 📊 Key Findings

1. ✅ **Consistent RL Improvement** – Both agents outperform baseline (+6% Q-Learning, +8% PPO)
2. 🧠 **Interpretable Policies** – Q-table shows learned state-action preferences
3. 🔄 **Live Learning Works** – Interactive demo proves real-time policy updates
4. 🎯 **Reward Design Matters** – Multi-component feedback enables nuanced learning

---

## 👥 Team

**The Rewards Musketeers**

This project was developed as part of an AI/RL course demonstrating practical applications of reinforcement learning to improve LLM-based systems.

---

## 🙏 Acknowledgements

- **Groq** for fast LLM inference (Llama-3.1-8b-instant)
- **OpenAI** for GPT-4o-mini judge evaluation
- **ChromaDB** for vector database infrastructure
- **Gradio** for interactive demo interface

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with:** Python • PyTorch • Groq • ChromaDB • Gradio  
**RL Algorithms:** Q-Learning • Proximal Policy Optimization (PPO)  
**Tested on:** Python 3.10.11, Windows 11 & macOS (Apple Silicon)
