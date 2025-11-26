# PCA-Agent Production Deployment Guide

## Quick Start with Docker

### 1. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Services will be available at:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - MLflow: http://localhost:5000
```

### 2. Run Individual Services

#### API Server
```bash
# Using Docker
docker run -p 8000:8000 -v $(pwd)/output:/app/output pca-agent python api.py

# Or directly
python api.py
# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

#### Streamlit Dashboard
```bash
# Using Docker
docker run -p 8501:8501 -v $(pwd)/output:/app/output pca-agent streamlit run dashboard.py

# Or directly
streamlit run dashboard.py
# Access at: http://localhost:8501
```

#### MLflow Tracking Server
```bash
# Using Docker
docker run -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns pca-agent mlflow server --host 0.0.0.0

# Or directly
mlflow server --host 0.0.0.0 --port 5000
# Access at: http://localhost:5000
```

## Installation

### Option 1: Docker (Recommended)
```bash
# Build image
docker build -t pca-agent .

# Run container
docker run -p 8000:8000 pca-agent
```

### Option 2: Local Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# Install production dependencies
pip install -r requirements-prod.txt
```

## Usage Examples

### 1. REST API

```python
import requests

# Upload and analyze campaign data
with open('campaign_data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f},
        data={
            'target_column': 'conversions',
            'tune_hyperparameters': 'false'
        }
    )

results = response.json()
print(f"Best Model: {results['best_model']}")
print(f"R² Score: {results['r2_score']}")

# Download results
response = requests.get('http://localhost:8000/download/model_results.csv')
with open('results.csv', 'wb') as f:
    f.write(response.content)
```

### 2. Database Integration

```python
from db_connector import DatabaseConnector
from main import run_pca_agent

# Connect to PostgreSQL
db = DatabaseConnector('postgresql://user:password@localhost:5432/campaign_db')

# Load data
df = db.load_campaign_data(
    table_name='campaigns',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Save to CSV
df.to_csv('campaign_data.csv', index=False)

# Run analysis
results = run_pca_agent(
    data_path='campaign_data.csv',
    target_column='conversions'
)

# Save results back to database
db.save_results(results['results'], 'analysis_results')
```

### 3. Automated Scheduling

```python
# Create scheduler_config.json
{
    "database": {
        "connection_string": "postgresql://user:password@localhost:5432/campaign_db",
        "table_name": "campaigns",
        "results_table": "analysis_results"
    },
    "email": {
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "from": "pca-agent@company.com",
        "to": ["analyst@company.com"]
    },
    "target_column": "conversions",
    "schedule_time": "09:00"
}

# Run scheduler
python scheduler.py
```

### 4. MLflow Tracking

```python
from mlflow_tracker import MLflowTracker
from main import run_pca_agent

# Initialize tracker
tracker = MLflowTracker(experiment_name="Campaign-Analysis-2024")

# Run analysis
results = run_pca_agent(
    data_path='campaign_data.csv',
    target_column='conversions'
)

# Log to MLflow
tracker.log_campaign_analysis(
    model=results['model'],
    model_name=results['model_name'],
    results_df=results['results'],
    feature_importance=results['feature_importance'],
    config={'target_column': 'conversions'}
)

# View in MLflow UI
# mlflow ui
# Navigate to http://localhost:5000
```

### 5. Unit Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_pca_agent.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Production Deployment

### AWS Deployment

```bash
# 1. Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t pca-agent .
docker tag pca-agent:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/pca-agent:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/pca-agent:latest

# 2. Deploy to ECS/Fargate
# Use AWS Console or CLI to create ECS service
```

### Google Cloud Deployment

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/pca-agent

# 2. Deploy to Cloud Run
gcloud run deploy pca-agent \
    --image gcr.io/<project-id>/pca-agent \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Azure Deployment

```bash
# 1. Build and push to ACR
az acr build --registry <registry-name> --image pca-agent .

# 2. Deploy to Azure Container Instances
az container create \
    --resource-group <resource-group> \
    --name pca-agent \
    --image <registry-name>.azurecr.io/pca-agent \
    --ports 8000
```

## Environment Variables

```bash
# Database
export DB_CONNECTION_STRING="postgresql://user:password@localhost:5432/campaign_db"

# Email
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export EMAIL_USERNAME="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"

# MLflow
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Application
export TARGET_COLUMN="conversions"
export TUNE_HYPERPARAMETERS="false"
```

## Monitoring & Logging

### View Logs
```bash
# Docker logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Application logs
tail -f scheduler.log
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Dashboard health
curl http://localhost:8501/healthz
```

## Troubleshooting

### Common Issues

1. **Port already in use**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
```

2. **Database connection failed**
```bash
# Test connection
python -c "from db_connector import DatabaseConnector; db = DatabaseConnector('your-connection-string'); db.connect()"
```

3. **Out of memory**
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory
```

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- Run tests: `pytest tests/ -v`
- Review documentation: `README.md`
