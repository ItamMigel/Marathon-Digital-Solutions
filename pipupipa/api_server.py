from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging
import json
import time
import datetime
import uvicorn
import os
import threading
from realtime_model import (
    initialize_system, predict_for_site, get_site_status,
    get_all_sites_status, get_feature_importance, update_model
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    filename='api_server.log',
    filemode='a'
)
logger = logging.getLogger()

# Инициализация FastAPI
app = FastAPI(
    title="Granulometry Monitoring API",
    description="API для мониторинга гранулометрии в реальном времени",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных для API
class PredictionRequest(BaseModel):
    site_id: str
    features: Dict[str, Any]
    timestamp: Optional[str] = None

class UpdateDataItem(BaseModel):
    features: Dict[str, float]
    granulometry: float

class UpdateModelRequest(BaseModel):
    site_id: str
    data: List[UpdateDataItem]

class SiteConfigUpdate(BaseModel):
    name: Optional[str] = None
    target_range: Optional[tuple] = None

# Глобальные переменные
api_requests_counter = 0
api_start_time = time.time()

# Middleware для подсчета запросов и логирования
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global api_requests_counter
    
    # Увеличиваем счетчик запросов
    api_requests_counter += 1
    
    # Логируем запрос
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Замеряем время выполнения
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Логируем информацию о времени выполнения
    logger.info(f"Response time: {process_time:.4f} seconds")
    
    # Добавляем заголовок с временем обработки
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Роуты API
@app.get("/")
async def root():
    """Корневой эндпоинт, возвращает информацию об API"""
    return {
        "name": "Granulometry Monitoring API",
        "version": "1.0.0",
        "status": "online",
        "uptime": int(time.time() - api_start_time)
    }

@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "requests_processed": api_requests_counter,
        "uptime_seconds": int(time.time() - api_start_time)
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Эндпоинт для получения предсказания гранулометрии"""
    try:
        # Проверяем наличие site_id
        if not request.site_id:
            raise HTTPException(status_code=400, detail="Site ID is required")
        
        # Преобразуем данные в DataFrame
        data = pd.DataFrame([request.features])
        
        # Выполняем предсказание
        result = predict_for_site(request.site_id, data)
        
        # Если есть ошибка, возвращаем ее
        if 'error' in result and not 'prediction' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sites")
async def get_sites():
    """Получение информации о всех участках"""
    try:
        result = get_all_sites_status()
        return result
    except Exception as e:
        logger.error(f"Error in get_sites endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sites/{site_id}")
async def get_site(site_id: str):
    """Получение информации о конкретном участке"""
    try:
        result = get_site_status(site_id)
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_site endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sites/{site_id}/importance")
async def get_importance(site_id: str):
    """Получение важности признаков для конкретного участка"""
    try:
        result = get_feature_importance(site_id)
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_importance endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sites/{site_id}/update")
async def update_site_model(site_id: str, request: UpdateModelRequest, background_tasks: BackgroundTasks):
    """Обновление модели для участка с новыми данными"""
    try:
        # Проверяем наличие данных
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Преобразуем данные в DataFrame
        features_list = []
        granulometry_list = []
        
        for item in request.data:
            features_list.append(item.features)
            granulometry_list.append(item.granulometry)
        
        # Создаем DataFrame
        df = pd.DataFrame(features_list)
        df['Гранулометрия %'] = granulometry_list
        
        # Запускаем обновление модели в фоновом режиме
        background_tasks.add_task(update_model, site_id, df)
        
        return {
            "status": "success",
            "message": f"Model update for site {site_id} started",
            "records": len(request.data),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in update_site_model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sites/{site_id}/config")
async def update_site_config(site_id: str, config: SiteConfigUpdate):
    """Обновление конфигурации участка"""
    try:
        from realtime_model import site_configs
        
        # Проверяем наличие участка
        if site_id not in site_configs:
            raise HTTPException(status_code=404, detail=f"Site {site_id} not found")
        
        # Обновляем конфигурацию
        updated = False
        
        if config.name is not None:
            site_configs[site_id].name = config.name
            updated = True
        
        if config.target_range is not None:
            site_configs[site_id].target_range = config.target_range
            updated = True
        
        if not updated:
            return {
                "status": "warning",
                "message": "No changes provided"
            }
        
        return {
            "status": "success",
            "message": f"Configuration for site {site_id} updated",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in update_site_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Получение статистики использования API"""
    return {
        "requests_processed": api_requests_counter,
        "uptime_seconds": int(time.time() - api_start_time),
        "start_time": datetime.datetime.fromtimestamp(api_start_time).isoformat(),
        "current_time": datetime.datetime.now().isoformat()
    }

# Обработчик ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Инициализация системы при старте сервера
@app.on_event("startup")
async def startup_event():
    """Действия при старте сервера"""
    logger.info("Starting API server")
    initialize_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Действия при остановке сервера"""
    logger.info("Shutting down API server")

# Запуск сервера если запускаем файл напрямую
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 