from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, inspect, Column, Integer, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import datetime
import logging
import asyncio
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("production-api")

# Database configuration
DATABASE_URL = "sqlite:///../value.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# FastAPI app
app = FastAPI(
    title="Production Monitoring API",
    description="API for monitoring production line data",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection context manager
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

# Database session dependency
def get_db_session():
    with get_db() as session:
        yield session

# Schemas
class TableInfo(BaseModel):
    name: str
    columns: List[str]

class TableData(BaseModel):
    table_name: str
    columns: List[str]
    data: List[Dict[str, Any]]
    total_records: int

class AnalysisResult(BaseModel):
    table_name: str
    production_line: Optional[int] = None
    summary: Dict[str, Any]
    recommendations: List[str]

# Dynamic model cache
model_cache = {}

# Get dynamic model for a table
def get_table_model(table_name: str):
    if table_name in model_cache:
        return model_cache[table_name]
    
    # Inspect the table to get column information
    inspector = inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns(table_name)}
    
    if not columns:
        return None
    
    # Create a dynamic model
    class DynamicTable(Base):
        __tablename__ = table_name
        __table_args__ = {"extend_existing": True}
        
        # Add columns dynamically based on database inspection
        for col_name, col_info in columns.items():
            locals()[col_name] = Column(
                col_info["type"], 
                primary_key=col_info.get("primary_key", False)
            )
    
    model_cache[table_name] = DynamicTable
    return DynamicTable

# API routes
@app.get("/")
def read_root():
    return {"message": "Production Monitoring API is running"}

@app.get("/api/tables", response_model=List[TableInfo])
def get_tables(db: Session = Depends(get_db_session)):
    """Get all available tables in the database"""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    tables = []
    for table_name in table_names:
        columns = [column["name"] for column in inspector.get_columns(table_name)]
        tables.append(TableInfo(name=table_name, columns=columns))
    
    return tables

@app.get("/api/tables/{table_name}", response_model=TableData)
def get_table_data(
    table_name: str,
    db: Session = Depends(get_db_session),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    columns: Optional[str] = None,
    filters: Optional[str] = None,
):
    """Get data from a specific table with pagination and filtering"""
    # Get table model
    table_model = get_table_model(table_name)
    if not table_model:
        return {"error": f"Table {table_name} not found"}
    
    # Get column information
    inspector = inspect(engine)
    all_columns = [column["name"] for column in inspector.get_columns(table_name)]
    
    # Filter selected columns or use all
    selected_columns = all_columns
    if columns:
        col_list = columns.split(",")
        selected_columns = [col for col in col_list if col in all_columns]
    
    # Base query
    query = db.query(table_model)
    
    # Apply filters if provided
    if filters:
        try:
            filter_parts = filters.split(",")
            for part in filter_parts:
                if ":" in part and "=" in part:
                    col_name, operator, value = part.split(":")
                    if col_name in all_columns:
                        column = getattr(table_model, col_name)
                        if operator == "eq":
                            query = query.filter(column == value)
                        elif operator == "gt":
                            query = query.filter(column > float(value))
                        elif operator == "lt":
                            query = query.filter(column < float(value))
                        elif operator == "contains":
                            query = query.filter(column.like(f"%{value}%"))
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
    
    # Get total count for pagination
    total_count = query.count()
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    # Execute query
    results = query.all()
    
    # Convert results to dict
    data = []
    for row in results:
        row_dict = {}
        for col in selected_columns:
            row_dict[col] = getattr(row, col)
        data.append(row_dict)
    
    return TableData(
        table_name=table_name,
        columns=selected_columns,
        data=data,
        total_records=total_count
    )

@app.get("/api/analysis/{table_name}", response_model=AnalysisResult)
def get_analysis(
    table_name: str, 
    production_line: Optional[int] = None,
    db: Session = Depends(get_db_session)
):
    """Get analysis results for a specific table/production line"""
    # Placeholder for actual analysis logic
    # In a real application, this would perform actual data analysis
    
    summary = {
        "average_values": {
            "power": 125.7,
            "temperature": 85.2,
            "efficiency": 78.9
        },
        "trends": {
            "power_trend": "increasing",
            "temperature_trend": "stable",
            "efficiency_trend": "decreasing"
        },
        "anomalies_detected": 3
    }
    
    recommendations = [
        "Consider maintenance for production line due to efficiency decrease",
        "Monitor power consumption trends over next 24 hours",
        "Temperature readings are within normal parameters"
    ]
    
    return AnalysisResult(
        table_name=table_name,
        production_line=production_line,
        summary=summary,
        recommendations=recommendations
    )

# Background task to periodically refresh data cache
async def refresh_data_cache():
    """Background task to refresh data cache every 10 minutes"""
    while True:
        try:
            logger.info("Refreshing data cache...")
            # In a real application, this would update any cached data
            
            # Reset model cache to ensure fresh schema
            model_cache.clear()
            
            logger.info("Data cache refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing data cache: {str(e)}")
        
        # Wait for 10 minutes before refreshing again
        await asyncio.sleep(600)  # 600 seconds = 10 minutes

@app.on_event("startup")
async def startup_event():
    """Start background tasks when application starts"""
    # Start the data refresh task
    asyncio.create_task(refresh_data_cache())
    logger.info("Application started, background tasks initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 