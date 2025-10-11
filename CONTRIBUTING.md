# Contributing to RL GHG Consultant

Thank you for your interest in contributing to the RL GHG Consultant project!

## How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on GitHub
- Clone your fork locally

### 2. Set Up Development Environment
```bash
git clone <your-fork-url>
cd RL_2025
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 3. Make Changes
- Create a new branch: `git checkout -b feature/your-feature-name`
- Make your changes
- Test thoroughly
- Update documentation if needed

### 4. Submit Pull Request
- Commit your changes: `git commit -m "Add feature: description"`
- Push to your fork: `git push origin feature/your-feature-name`
- Create a Pull Request on GitHub

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Add docstrings to functions and classes

### Testing
- Test your changes with the interactive demo: `python three_bot_demo.py`
- Run experiments: `python complete_experiment.py`
- Ensure all tests pass

### Documentation
- Update `docs/STUDY.md` for technical changes
- Update `README.md` for user-facing changes
- Add comments for complex logic

## Areas for Contribution

### 🐛 Bug Fixes
- Fix any issues you encounter
- Improve error handling
- Optimize performance

### 🚀 Features
- New RL algorithms
- Additional evaluation metrics
- UI/UX improvements
- New retrieval strategies

### 📚 Documentation
- Improve tutorials
- Add examples
- Translate documentation

### 🧪 Research
- Experiment with new approaches
- Compare different RL methods
- Analyze performance patterns

## Questions?

Feel free to open an issue for questions or discussions!

---

**Thank you for contributing to RL-enhanced AI systems!** 🌱
