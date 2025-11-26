"""
FastAPI REST API for PCA-Agent
Provides HTTP endpoints for campaign analysis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os
import tempfile
from datetime import datetime

from main import run_pca_agent

app = FastAPI(
    title="PCA-Agent API",
    description="Post-Campaign Analysis AI Agent API",
    version="1.0.0"
)


class AnalysisRequest(BaseModel):
    """Request model for analysis"""
    target_column: str = "conversions"
    tune_hyperparameters: bool = False
    models_to_train: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    status: str
    best_model: str
    r2_score: float
    rmse: float
    mae: float
    mape: float
    execution_time: float
    output_files: List[str]


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PCA-Agent API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": 15
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_campaign(
    file: UploadFile = File(...),
    target_column: str = "conversions",
    tune_hyperparameters: bool = False
):
    """
    Analyze campaign data from uploaded CSV file
    
    Args:
        file: CSV file with campaign data
        target_column: Target variable to predict
        tune_hyperparameters: Whether to tune model hyperparameters
        
    Returns:
        Analysis results with model performance metrics
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run analysis
        start_time = datetime.now()
        
        results = run_pca_agent(
            data_path=tmp_path,
            target_column=target_column,
            use_sample_data=False,
            tune_hyperparameters=tune_hyperparameters
        )
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Prepare response
        best_result = results['results'].iloc[0]
        
        return AnalysisResponse(
            status="success",
            best_model=results['model_name'],
            r2_score=float(best_result['R2']),
            rmse=float(best_result['RMSE']),
            mae=float(best_result['MAE']),
            mape=float(best_result['MAPE']),
            execution_time=execution_time,
            output_files=[
                "output/model_results.csv",
                "output/feature_importance.csv",
                "output/executive_summary.txt"
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename}")
async def download_output(filename: str):
    """
    Download generated output files
    
    Args:
        filename: Name of the file to download
        
    Returns:
        File download response
    """
    file_path = f"output/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/models")
async def list_models():
    """List all available regression models"""
    from src.config import MODELS_TO_TRAIN
    
    return {
        "available_models": MODELS_TO_TRAIN,
        "total_count": len(MODELS_TO_TRAIN)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
