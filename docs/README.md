# 📚 RL GHG Consultant - Documentation

This directory contains comprehensive documentation for the RL GHG Consultant project.

## 📖 Contents

### **Core Documentation**
- **[STUDY.md](./STUDY.md)** - Complete technical study guide (2,350+ lines)
  - Backend files explanation
  - RL environment and Q-Learning strategy
  - Reward function details
  - Thumbs up/down mechanism
  - Project setup and usage guide
  - Troubleshooting and optimization

### **Visual Results**
- **[images/](./images/)** - Charts and comparison graphs
  - `complete_comparison_3methods.png` - Full experiment results dashboard
  - `q_vs_ppo_comparison.png` - Q-Learning vs PPO comparison

### **Results Archive**
- **[results/](./results/)** - Detailed experiment results (future)

## 🎯 Quick Navigation

### **For Understanding the System:**
1. Read [STUDY.md](./STUDY.md) sections 1-4 (Backend, RL Environment, Reward Function)
2. Look at [images/complete_comparison_3methods.png](./images/complete_comparison_3methods.png) for results

### **For Setup & Usage:**
1. Read [STUDY.md](./STUDY.md) section 5 (Project Setup & Usage Guide)
2. Follow the installation steps

### **For Troubleshooting:**
1. Check [STUDY.md](./STUDY.md) Troubleshooting section
2. Look at common issues and solutions

## 📊 Key Results Summary

From the dashboard comparison:

| Method | Average Score | Thumbs Up Rate | Improvement |
|--------|---------------|----------------|-------------|
| **Baseline** | 0.840 | 100.0% | - |
| **Q-Learning** | 0.880 | 90.0% | +4.8% |
| **PPO** | 0.840 | 90.0% | +0.0% |

**Key Finding:** Q-Learning shows clear improvement over baseline (+4.8%), while PPO matches baseline performance. Q-Learning demonstrates the most consistent learning!

---

**Last Updated:** October 12, 2025
