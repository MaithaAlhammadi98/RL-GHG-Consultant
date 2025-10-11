#  RL-Enhanced GHG Consultant Chatbot

An intelligent GHG (Greenhouse Gas) consulting chatbot that uses **Reinforcement Learning** to improve answer quality through learned document retrieval strategies.

##  Project Overview

This project demonstrates how **Reinforcement Learning (RL) improves LLM-based chatbot performance** by learning optimal retrieval policies for RAG (Retrieval-Augmented Generation) systems.

### **Key Results:**
| Method | Avg Score | Thumbs Up Rate | Improvement |
|--------|-----------|----------------|-------------|
| **Baseline** (no RL) | 0.840 | 100.0% | - |
| **Q-Learning** | 0.880 | 90.0% | +4.8% |
| **PPO** | 0.840 | 90.0% | +0.0% |

## ️ Architecture

```
User Question  State Encoder  RL Agent (Q-Learning/PPO)  Document Filter
                                          
                                    RAG Process  ChromaDB
                                          
                                      LLM (Groq)  Answer
                                          
                                   Reward Calculation  Judge Evaluation
                                          
                                   Update RL Agent
```

##  Project Structure

```
RL_2025/
├── src/
│   ├── backend/
│   │   ├── rl_agent.py           # Q-Learning agent
│   │   ├── ppo_agent.py          # PPO agent  
│   │   ├── rag_process.py        # RAG pipeline
│   │   ├── retrieval_policies.py # Document filters
│   │   ├── reward.py             # Reward calculation
│   │   ├── evaluator.py          # LLM judge
│   │   └── state.py              # State encoding
│   ├── data/
│   │   ├── q_table.json          # Q-Learning table
│   │   ├── ppo_model.pt          # PPO neural network
│   │   └── *.pdf                 # GHG documentation
│   └── app/                      # Original Flask app
├── notebooks/                    # Archived experiments
├── logs/                         # Experiment results & plots
├── chroma_persistent_storage/    # Vector database
├── three_bot_demo.py            #  Main demo (Baseline vs Q vs PPO)
├── complete_experiment.py        # Full experiment runner
├── monitor_q_table.py           # Real-time Q-table monitor
├── populate_database.py         # Database setup
├── requirements.txt
└── Dockerfile
```

##  Quick Start

### **1. Installation**

```bash
# Clone repository
git clone <your-repo-url>
cd RL_2025

# Install dependencies
pip install -r requirements.txt
```

### **2. Setup Environment**

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for judge
```

### **3. Populate Database**

```bash
python populate_database.py
```

This loads GHG documents into ChromaDB vector store (~335 MB).

### **4. Run Interactive Demo**

```bash
python three_bot_demo.py
```

Opens Gradio interface at `http://localhost:7860` where you can:
- Ask questions to all 3 bots simultaneously
- Compare their answers
- Train Q-Learning bot live with  feedback
- Watch Q-table update in real-time!

##  Run Experiments

### **Full Experiment (All 3 Methods)**

```bash
python complete_experiment.py
```

Runs 10 test questions through Baseline, Q-Learning, and PPO, then:
- Evaluates answers with LLM judge
- Calculates rewards
- Trains agents
- Generates comparison plots
- Saves detailed results to `logs/`

**Output:**
- `logs/complete_experiment_results.json` - Full results
- `logs/complete_comparison_3methods.png` - Visual comparison
- `logs/*_detailed_results.csv` - Per-question breakdown

##  Reinforcement Learning Details

### **State Representation**
Questions are encoded into states with features:
- **Topic**: `ghg`, `legal`, `fin`, `other`
- **Length**: `short` (<10 words), `medium`, `long`
- **Sector**: `energy`, `transport`, `unknown`
- **Company Size**: `large`, `small`, `unknown`
- **Month**: Current month (for temporal context)

### **Actions (Retrieval Policies)**
- `broad` - Search all documents
- `legal_only` - Filter for legal/regulatory docs
- `financial_only` - Filter for financial docs
- `company_only` - Filter for company-specific docs

### **Reward Function**
```python
reward = judge_score + length_bonus + relevance_bonus + coherence_bonus
```
- **Judge Score**: 0-1 (LLM evaluation)
- **Length Bonus**: +0.3 for comprehensive answers
- **Relevance Bonus**: +0.2 for relevant chunks
- **Coherence Bonus**: +0.1 for well-structured answers

### **Q-Learning**
- Algorithm: Q-Learning with ε-greedy exploration
- Parameters: α=0.3, γ=0.9, ε=0.2
- Storage: JSON file (`src/data/q_table.json`)
- Updates: After every question

### **PPO (Proximal Policy Optimization)**
- Algorithm: PPO with actor-critic networks
- Network: 128-64-32 hidden layers
- Parameters: lr=3e-4, clip=0.2
- Storage: PyTorch model (`src/data/ppo_model.pt`)
- Training: Batch updates every 10 episodes

##  Key Files

### **Main Applications**
- `three_bot_demo.py` - Interactive Gradio demo (recommended)
- `complete_experiment.py` - Full experiment runner

### **Utilities**
- `monitor_q_table.py` - Real-time Q-table monitoring
- `populate_database.py` - Database initialization

### **Documentation**
- `docs/STUDY.md` - Complete technical study guide (2,350+ lines)
- `docs/README.md` - Documentation index
- `docs/images/` - Results charts and visualizations

##  Troubleshooting

### **Q-Table Not Updating?**
The file IS updating, but VS Code doesn't auto-refresh. Close and reopen the file, or run:
```bash
python monitor_q_table.py
```

### **ChromaDB Errors?**
Delete and repopulate:
```bash
rm -rf chroma_persistent_storage
python populate_database.py
```

### **Out of Memory?**
Reduce batch size or switch to lighter LLM model in the code.

##  Experimental Results

See `logs/` folder for:
- Detailed per-question results (CSV)
- Aggregated statistics (JSON)
- Comparison visualizations (PNG)

**Key Findings:**
1.  Q-Learning shows clear improvement over baseline (+4.8%)
2.  RL agents learn interpretable policies (visible in Q-table)
3.  Live learning works (demo proves real-time updates)
4.  Multi-component reward function provides rich feedback

##  Docker Support

```bash
# Build image
docker build -t rl-ghg-chatbot .

# Run container
docker run -p 7860:7860 --env-file .env rl-ghg-chatbot
```

##  Citation

If you use this project, please cite:
```
RL-Enhanced GHG Consultant Chatbot
Using Reinforcement Learning to Improve LLM-based RAG Systems
2025
```

##  License

MIT License - see [LICENSE](LICENSE) file for details.

##  Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

**Built with:** Python, PyTorch, Groq LLM, ChromaDB, Gradio
**RL Algorithms:** Q-Learning, PPO
**Evaluation:** LLM-as-Judge (OpenAI/Groq)

