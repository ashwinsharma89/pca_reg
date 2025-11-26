# Dockerfile for PCA-Agent
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    streamlit \
    schedule \
    pytest \
    mlflow

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p output

# Expose ports
EXPOSE 8000 8501 5000

# Default command (can be overridden)
CMD ["python", "main.py"]
