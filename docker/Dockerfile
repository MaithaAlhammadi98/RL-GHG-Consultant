# ============================================================================
# RL-Enhanced GHG Consultant - Docker Container
# ============================================================================
# Multi-stage build for optimized production deployment
# ============================================================================

# Stage 1: Base Python environment
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Stage 3: Application setup
FROM dependencies as app

# Copy application code from parent directory
COPY ../ .

# Create necessary directories
RUN mkdir -p logs/baseline logs/qlearning logs/ppo logs/comparisons src/data

# Set proper permissions
RUN chmod +x *.py

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "three_bot_demo.py"]
