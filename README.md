<p align="center">
  <img src="docs/images/banner.jpg" width="100%" alt="RL-GHG-Consultant Banner">
</p>

# 🌍 RL-GHG-Consultant
> **Fine-Tuning Language Models through Reinforcement Learning for GHG Compliance**  
> *By The Rewards Musketeers*  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey)]()

---

## 🚀 Overview
An intelligent chatbot designed for Greenhouse Gas (GHG) compliance, powered by **Reinforcement Learning** within a **Retrieval-Augmented Generation (RAG)** framework.  
Our RL agents (**Q-Learning** and **PPO**) dynamically choose optimal document filters—legal, financial, or technical—improving response quality by **6–8%**.

> **Innovation:** A multi-component reward function that evaluates answer quality, retrieval relevance, grounding, and policy diversity for continuous learning and interpretability.

---

## 🧠 Architecture

```
User Question → RL Agent (Q-Learning / PPO) → Document Filter
                            ↓
                      RAG + ChromaDB
                            ↓
                     LLM (Groq) → Answer
                            ↓
                Reward Calculation ← LLM Judge
```

---

## 🎯 Results

| Method | Avg Judge Score | Δ Improvement | 👍 Feedback |
|--------|-----------------|---------------|-------------|
| **Baseline** | 0.83 | — | 100% |
| **Q-Learning** | 0.88 | +6% | 90% |
| **PPO** | 0.90 | +8.4% | 100% |

> *(Evaluated on 40 test questions using GPT-4o-mini as judge)*  
> 🔍 [Full Experiment Logs & Visuals](docs/STUDY.md)

---

## 🧩 RL Components Deep Dive

### **State Space Definition**

Each user question is encoded into a structured state using [`src/backend/state.py`](src/backend/state.py):

| Feature | Values | Purpose |
|---------|--------|---------|
| **Topic** | `legal`, `fin`, `ghg`, `other` | Question category via regex matching |
| **Length** | `short` (<80 chars), `medium` (80-200), `long` (>200) | Question complexity proxy |
| **Sector** | `energy`, `transport`, `finance`, `unknown` | Company context |
| **Size** | `small`, `medium`, `large`, `unknown` | Company scale |
| **Month** | `YYYY-MM` | Temporal context for seasonal patterns |

**Example State:**
```python
{
  "topic": "ghg",
  "len": "medium", 
  "sector": "energy",
  "size": "large",
  "month": "2025-10"
}
```

This **low-dimensional, interpretable encoding** allows Q-Learning to build explicit state-action mappings while remaining human-readable. PPO's neural network can learn higher-order feature interactions from these base features.

### **Action Space**

- `broad` – Search all documents (default exploration)
- `legal_only` – Filter for regulatory/compliance documents
- `financial_only` – Filter for ESG/financial reports  
- `company_only` – Filter for company-specific documents

### **Reward Function**

```python
total_reward = 0.5*judge_score + 0.2*retrieval_score \
             + 0.15*action_score + 0.15*grounding_score
```

---

## ⚙️ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/MaithaAlhammadi98/RL-GHG-Consultant.git
cd RL-GHG-Consultant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.example .env  # Add your GROQ_API_KEY & OPENAI_API_KEY

# 4. Download pre-built database (133 MB - saves 10-15 minutes)
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/datasets/petitkitten/rl-ghg-consultant-database/resolve/main/chroma_database_v1.0.zip" -OutFile "chroma_database.zip"
Expand-Archive -Path "chroma_database.zip" -DestinationPath "." -Force

# macOS/Linux
wget https://huggingface.co/datasets/petitkitten/rl-ghg-consultant-database/resolve/main/chroma_database_v1.0.zip
unzip chroma_database_v1.0.zip

# 5. Run interactive demo
python three_bot_demo.py  # Opens at http://localhost:7860
```

> **Note:** To build the database from scratch instead (10-15 min), run [`python src/backend/populate_database.py`](src/backend/populate_database.py)

---

## 🧩 Interactive RL Demo Interface

<p align="center">
  <img src="docs/images/DEMO.png" width="85%" alt="Interactive RL Demo Interface">
</p>
<p align="center">
  <img src="docs/images/DEMO_RESULTS.png" width="85%" alt="Interactive RL Demo Interface">
</p>

**Features:**
- 🤖 Compare Baseline, Q-Learning, and PPO bots side-by-side
- 👍👎 Provide live feedback to train Q-Learning agent in real-time
- 📊 Watch Q-table values update as the agent learns
- 🎮 Interactive policy exploration with immediate visual feedback


---

## 📊 Key Findings

* ✅ RL agents outperform static retrieval by +6–8%
* 🧠 Q-tables show interpretable state-action patterns
* 🔄 Live feedback improves policies in real-time
* 🎯 Reward shaping enables nuanced, adaptive learning

<p align="center">
  <img src="docs/images/complete_comparison_3methods.png" width="85%" alt="RL Three-Bot Comparison Results">
</p>

---

## 📁 Project Structure

```
RL-GHG-Consultant/
├── src/backend/              # Core RL & RAG implementation
│   ├── rl_agent.py          # Q-Learning agent
│   ├── ppo_agent.py         # PPO agent
│   ├── rag_process.py       # RAG pipeline
│   ├── reward_enhanced.py   # Multi-component reward
│   ├── state.py             # State encoder
│   └── populate_database.py # Database generation script
├── three_bot_demo.py        # 🎮 Interactive Gradio demo
├── complete_experiment.py   # 📊 Full experiment runner
├── docs/                    # 📚 Documentation
│   ├── STUDY.md            # Complete technical guide
│   └── images/             # Visualizations
├── logs/                    # 📈 Experiment results
└── requirements.txt         # Python dependencies
```

---

## 👥 Team

**The Rewards Musketeers**  
Developed for the UTS Reinforcement Learning course, showcasing real-world RL for LLM optimization.

---

## 🙏 Acknowledgements

**Groq** • **OpenAI** • **ChromaDB** • **Gradio**

---

## 📚 Documentation

### **Guides & Reports**
- 📖 **[Complete Technical Study Guide](docs/STUDY.md)** – Full architecture, design decisions, experiments (2,350+ lines)
- 📊 **[Experiment Results](logs/comparisons/)** – Detailed CSV/JSON logs and visualizations  
- 🎓 **[Project Report](REPORT.md)** – Academic report (to be uploaded)

### **Main Scripts**
- 🎮 **[`three_bot_demo.py`](three_bot_demo.py)** – Interactive Gradio demo (3 bots with live feedback)
- 📊 **[`complete_experiment.py`](complete_experiment.py)** – Full experiment runner (train & evaluate all methods)
- 🗄️ **[`src/backend/populate_database.py`](src/backend/populate_database.py)** – Database generation from PDFs

### **Core RL Components**
- 🎯 **[`src/backend/rl_agent.py`](src/backend/rl_agent.py)** – Q-Learning agent implementation
- 🧠 **[`src/backend/ppo_agent.py`](src/backend/ppo_agent.py)** – PPO agent with actor-critic network
- 🔍 **[`src/backend/rag_process.py`](src/backend/rag_process.py)** – RAG pipeline with ChromaDB
- 🎁 **[`src/backend/reward_enhanced.py`](src/backend/reward_enhanced.py)** – Multi-component reward function
- 🔢 **[`src/backend/state.py`](src/backend/state.py)** – State encoding for RL agents

---

## 💡 Areas for Future Improvement

### **1. Enhanced State Representation**

**Current:** Low-dimensional manual feature engineering (topic, length, sector, size, month)

**Future Directions:**
- **Semantic Embeddings**: Include question embedding vectors (e.g., from `sentence-transformers`) for richer state representation
- **Historical Context**: Track dialogue history and previous action success rates per user session
- **Entity Recognition**: Extract named entities (company names, regulations, dates) as additional state features
- **Dynamic Features**: Include real-time metrics like retrieval latency, chunk availability per filter

### **2. Scaling the Action Space**

**Current:** Fixed 4-action space (`broad`, `legal_only`, `financial_only`, `company_only`)

**Expansion Strategies:**

**For Q-Learning:**
- **Nested Filters**: Introduce hierarchical actions (e.g., `legal_eu_regulations`, `legal_us_epa`, `financial_esg_metrics`)
- **Multi-Select Actions**: Allow combinations like `[legal + financial]` for cross-domain queries
- **Confidence Thresholds**: Add actions with varying retrieval strictness (e.g., `top-3-strict`, `top-10-relaxed`)

**For PPO (Advanced):**
- **Continuous Action Space**: Train PPO to output a probability distribution over document types (e.g., 50% Legal, 30% Financial, 20% Technical)
- **Weighted Retrieval**: Use action outputs as query weights for mixed-source retrieval, enabling nuanced, multi-faceted responses
- **Parameterized Actions**: Learn continuous parameters like `temperature`, `top_k`, `similarity_threshold` for retrieval

### **3. Production Deployment Strategy**

**Current:** Side-by-side comparison demo with live Q-Learning updates

**Deployment Considerations:**

**Q-Learning in Production:**
- ✅ **Strengths**: Fast online learning from user feedback (👍👎), interpretable Q-table, minimal compute
- ⚠️ **Challenges**: Requires frequent user interactions, may converge slowly on rare states
- **Recommendation**: Deploy for interactive applications where users provide immediate feedback (customer support, internal tools)

**PPO in Production:**
- ✅ **Strengths**: Superior performance (+8.4% vs baseline), stable learning, handles complex state spaces
- ⚠️ **Challenges**: Requires batch training, not suitable for instant online updates
- **Recommendation**: Deployment workflow:
  1. **Initial Deployment**: Use pre-trained PPO model (from `complete_experiment.py`)
  2. **Data Collection**: Log user interactions (questions, actions, feedback) in production
  3. **Periodic Retraining**: Retrain PPO nightly/weekly on collected data using off-policy learning
  4. **A/B Testing**: Shadow mode comparison (PPO vs baseline) before full rollout
  5. **Model Versioning**: Maintain multiple PPO checkpoints for rollback safety

**Hybrid Approach (Best of Both Worlds):**
- Deploy PPO as primary agent for performance
- Run Q-Learning in parallel for exploration and cold-start scenarios
- Use Q-Learning feedback to identify distribution shifts for PPO retraining triggers

### **4. Advanced Evaluation**

**Future Metrics:**
- **User Engagement**: Session length, follow-up question rate, document click-through
- **Retrieval Efficiency**: Average chunks needed per satisfactory answer (lower is better)
- **Diversity**: Coverage of different document types and topics over time
- **Failure Analysis**: Automatic detection of low-confidence answers for human review queues

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

**Built with:** Python • PyTorch • Groq • ChromaDB • Gradio  
**RL Algorithms:** Q-Learning • Proximal Policy Optimization (PPO)  
**Tested on:** Python 3.10.11, Windows 11 & macOS (Apple Silicon)
