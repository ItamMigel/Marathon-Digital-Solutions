#!/usr/bin/env python3
"""
Realtime Model Manager for Granulometry Prediction System

This module provides functionality for managing and interacting with machine learning models
for multiple sites in real-time, including:
- Model loading and caching
- Site configuration management
- Prediction with confidence intervals
- Feature importance analysis
- Model retraining and updating
"""

import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import psutil
from threading import Lock
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("realtime_model.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("RealtimeModel")

# Global variables
MODEL_CACHE = {}  # Cache for loaded models
SITE_CONFIGS = {}  # Cache for site configurations
MODEL_LOCKS = {}  # Locks to prevent concurrent model updates
STATS = {
    "start_time": time.time(),
    "total_predictions": 0,
    "prediction_times": [],
    "recent_predictions": []
}

# Constants
MAX_CACHED_MODELS = 10
MAX_RECENT_PREDICTIONS = 100
MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
CONFIG_DIR = os.environ.get('CONFIG_DIR', 'configs')


class ModelManager:
    """Manager class for handling model operations for a specific site."""
    
    def __init__(self, site_id: str):
        """Initialize the model manager for a specific site.
        
        Args:
            site_id: Identifier for the site
        """
        self.site_id = site_id
        self.config = self._load_site_config()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.lock = MODEL_LOCKS.setdefault(site_id, Lock())
    
    def _load_site_config(self) -> Dict[str, Any]:
        """Load site configuration from file or cache."""
        if self.site_id in SITE_CONFIGS:
            return SITE_CONFIGS[self.site_id]
        
        config_path = os.path.join(CONFIG_DIR, f"{self.site_id}_config.json")
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found for site {self.site_id}. Using default configuration.")
            
            # Create a default configuration
            default_config = {
                "name": f"Site {self.site_id}",
                "status": "inactive",
                "model_path": os.path.join(MODELS_DIR, f"{self.site_id}_model.joblib"),
                "scaler_path": os.path.join(MODELS_DIR, f"{self.site_id}_scaler.joblib"),
                "feature_names_path": os.path.join(MODELS_DIR, f"{self.site_id}_features.json"),
                "target_range": [0, 100],
                "confidence_level": 0.95,
                "last_updated": datetime.now().isoformat(),
                "prediction_threshold": 0.75,
                "required_features": []
            }
            
            # Create directories if they don't exist
            os.makedirs(CONFIG_DIR, exist_ok=True)
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            # Save the default configuration
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            SITE_CONFIGS[self.site_id] = default_config
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            SITE_CONFIGS[self.site_id] = config
            return config
        except Exception as e:
            logger.error(f"Error loading configuration for site {self.site_id}: {str(e)}")
            raise ValueError(f"Failed to load site configuration: {str(e)}")
    
    def load_model(self) -> bool:
        """Load the model and associated artifacts from disk."""
        # If model is already in cache, return it
        if self.site_id in MODEL_CACHE:
            self.model, self.scaler, self.feature_names = MODEL_CACHE[self.site_id]
            return True
        
        # Check if model exists
        model_path = self.config["model_path"]
        scaler_path = self.config["scaler_path"]
        feature_names_path = self.config["feature_names_path"]
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_names_path]):
            logger.warning(f"Model artifacts not found for site {self.site_id}")
            return False
        
        try:
            with self.lock:
                logger.info(f"Loading model for site {self.site_id}...")
                
                # Load model artifacts
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                
                # Update cache (with LRU-like eviction if needed)
                if len(MODEL_CACHE) >= MAX_CACHED_MODELS:
                    # Remove least recently used model (first in the dict)
                    oldest_site = next(iter(MODEL_CACHE))
                    del MODEL_CACHE[oldest_site]
                
                MODEL_CACHE[self.site_id] = (self.model, self.scaler, self.feature_names)
                
                logger.info(f"Model loaded successfully for site {self.site_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error loading model for site {self.site_id}: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the loaded model.
        
        Args:
            features: Dictionary of feature name/value pairs
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        start_time = time.time()
        
        # Load model if needed
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": f"Model for site {self.site_id} could not be loaded"}
        
        try:
            # Check for required features
            if "required_features" in self.config and self.config["required_features"]:
                missing_features = [f for f in self.config["required_features"] if f not in features]
                if missing_features:
                    return {
                        "error": f"Missing required features: {', '.join(missing_features)}",
                        "site_id": self.site_id
                    }
            
            # Prepare features dataframe
            df = pd.DataFrame([features])
            
            # Keep only the features the model knows about
            available_features = [f for f in self.feature_names if f in df.columns]
            
            if not available_features:
                return {
                    "error": "No valid features provided for prediction",
                    "site_id": self.site_id,
                    "expected_features": self.feature_names
                }
            
            # Fill missing values with 0
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Select and order features according to model's expected features
            X = df[self.feature_names].copy()
            
            # Check for non-numeric columns and handle them
            non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"Non-numeric columns found and will be dropped: {non_numeric_cols}")
                X = X.select_dtypes(include=['number'])
            
            # Apply scaling
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Calculate confidence interval if model supports it
            confidence_interval = None
            confidence_level = self.config.get("confidence_level", 0.95)
            
            try:
                if hasattr(self.model, 'predict_proba'):
                    # For classifiers with predict_proba
                    probas = self.model.predict_proba(X_scaled)[0]
                    confidence = max(probas)
                    std_dev = np.sqrt(confidence * (1 - confidence))
                    margin = 1.96 * std_dev  # 95% confidence interval
                    confidence_interval = [prediction - margin, prediction + margin]
                
                elif hasattr(self.model, 'estimators_'):
                    # For ensemble models like Random Forest
                    predictions = [estimator.predict(X_scaled)[0] for estimator in self.model.estimators_]
                    std_dev = np.std(predictions)
                    margin = 1.96 * std_dev  # 95% confidence interval
                    confidence_interval = [prediction - margin, prediction + margin]
                    
                else:
                    # Default case - use a simple percentage-based margin
                    margin = prediction * 0.1  # 10% margin
                    confidence_interval = [prediction - margin, prediction + margin]
            
            except Exception as e:
                logger.warning(f"Error calculating confidence interval: {str(e)}")
                # Fallback
                confidence_interval = [prediction * 0.9, prediction * 1.1]
            
            # Ensure confidence interval is within target range
            target_range = self.config.get("target_range", [0, 100])
            confidence_interval[0] = max(confidence_interval[0], target_range[0])
            confidence_interval[1] = min(confidence_interval[1], target_range[1])
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            # Update statistics
            STATS["total_predictions"] += 1
            STATS["prediction_times"].append(prediction_time)
            
            # Keep only the last N prediction times
            if len(STATS["prediction_times"]) > 1000:
                STATS["prediction_times"] = STATS["prediction_times"][-1000:]
            
            # Track recent predictions
            recent_prediction = {
                "site_id": self.site_id,
                "prediction": float(prediction),
                "timestamp": datetime.now().isoformat()
            }
            
            STATS["recent_predictions"].append(recent_prediction)
            if len(STATS["recent_predictions"]) > MAX_RECENT_PREDICTIONS:
                STATS["recent_predictions"] = STATS["recent_predictions"][-MAX_RECENT_PREDICTIONS:]
            
            # Prepare result
            result = {
                "prediction": float(prediction),
                "confidence_interval": [float(ci) for ci in confidence_interval],
                "confidence_level": confidence_level,
                "prediction_time": prediction_time,
                "site_id": self.site_id,
                "model_version": self._get_model_version(),
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "features_provided": len(features),
                    "features_used": len(available_features)
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction for site {self.site_id}: {str(e)}")
            return {
                "error": f"Prediction error: {str(e)}",
                "site_id": self.site_id
            }
    
    def get_importance(self) -> Dict[str, Any]:
        """Get feature importance from the model."""
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": f"Model for site {self.site_id} could not be loaded"}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Map importance values to feature names
                importance_dict = {}
                for feature, importance in zip(self.feature_names, importances):
                    importance_dict[feature] = float(importance)
                
                return {
                    "feature_importances": importance_dict,
                    "site_id": self.site_id,
                    "model_version": self._get_model_version()
                }
            
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0]) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
                
                # Map importance values to feature names
                importance_dict = {}
                for feature, importance in zip(self.feature_names, importances):
                    importance_dict[feature] = float(importance)
                
                return {
                    "feature_importances": importance_dict,
                    "site_id": self.site_id,
                    "model_version": self._get_model_version()
                }
            
            else:
                return {
                    "error": "Model does not provide feature importance information",
                    "site_id": self.site_id
                }
        
        except Exception as e:
            logger.error(f"Error getting feature importance for site {self.site_id}: {str(e)}")
            return {
                "error": f"Error retrieving feature importance: {str(e)}",
                "site_id": self.site_id
            }
    
    def _get_model_version(self) -> str:
        """Get the model version based on file modification time."""
        try:
            mtime = os.path.getmtime(self.config["model_path"])
            dt = datetime.fromtimestamp(mtime)
            return dt.strftime("%Y%m%d-%H%M%S")
        except Exception:
            return "unknown"
    
    def update_configuration(self, new_config: Dict[str, Any]) -> bool:
        """Update the site configuration.
        
        Args:
            new_config: Dictionary with configuration values to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                # Load current config
                current_config = self._load_site_config()
                
                # Update with new values
                for key, value in new_config.items():
                    current_config[key] = value
                
                # Update last_updated timestamp
                current_config["last_updated"] = datetime.now().isoformat()
                
                # Save updated config
                config_path = os.path.join(CONFIG_DIR, f"{self.site_id}_config.json")
                with open(config_path, 'w') as f:
                    json.dump(current_config, f, indent=2)
                
                # Update cache
                SITE_CONFIGS[self.site_id] = current_config
                self.config = current_config
                
                logger.info(f"Configuration updated for site {self.site_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error updating configuration for site {self.site_id}: {str(e)}")
            return False
    
    def save_model(self, model, scaler, feature_names: List[str]) -> bool:
        """Save a new model and associated artifacts to disk.
        
        Args:
            model: The trained model object
            scaler: The feature scaler object
            feature_names: List of feature names
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.lock:
                # Ensure directories exist
                os.makedirs(MODELS_DIR, exist_ok=True)
                
                # Save model artifacts
                model_path = self.config["model_path"]
                scaler_path = self.config["scaler_path"]
                feature_names_path = self.config["feature_names_path"]
                
                # Create backup of existing model if it exists
                for path in [model_path, scaler_path, feature_names_path]:
                    if os.path.exists(path):
                        backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        os.rename(path, backup_path)
                
                # Save new model and artifacts
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                with open(feature_names_path, 'w') as f:
                    json.dump(feature_names, f)
                
                # Update model in cache
                MODEL_CACHE[self.site_id] = (model, scaler, feature_names)
                self.model = model
                self.scaler = scaler
                self.feature_names = feature_names
                
                # Update config
                self.update_configuration({
                    "status": "active",
                    "last_updated": datetime.now().isoformat()
                })
                
                logger.info(f"Model saved successfully for site {self.site_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error saving model for site {self.site_id}: {str(e)}")
            return False


# Module-level functions for external use

def get_model_manager(site_id: str) -> ModelManager:
    """Get a model manager instance for the specified site."""
    return ModelManager(site_id)

def predict_granulometry(site_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction for the specified site with the given features."""
    manager = get_model_manager(site_id)
    return manager.predict(features)

def get_site_info(site_id: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Get information about a specific site or all sites."""
    if site_id:
        try:
            manager = get_model_manager(site_id)
            
            # Basic site info
            info = {
                "id": site_id,
                "name": manager.config.get("name", f"Site {site_id}"),
                "status": manager.config.get("status", "unknown"),
                "last_updated": manager.config.get("last_updated", "never"),
                "target_range": manager.config.get("target_range", [0, 100]),
                "confidence_level": manager.config.get("confidence_level", 0.95)
            }
            
            # Check if model exists
            model_path = manager.config.get("model_path", "")
            if model_path and os.path.exists(model_path):
                info["model_exists"] = True
                info["model_size_mb"] = os.path.getsize(model_path) / (1024 * 1024)
                info["model_version"] = manager._get_model_version()
            else:
                info["model_exists"] = False
            
            # Add required features if available
            if "required_features" in manager.config:
                info["required_features"] = manager.config["required_features"]
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting info for site {site_id}: {str(e)}")
            return {"error": f"Failed to get site info: {str(e)}"}
    else:
        # Get all sites
        sites = {}
        
        # First check config directory for site configs
        os.makedirs(CONFIG_DIR, exist_ok=True)
        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith("_config.json"):
                site_id = filename.replace("_config.json", "")
                try:
                    sites[site_id] = get_site_info(site_id)
                except Exception as e:
                    logger.error(f"Error getting info for site {site_id}: {str(e)}")
                    sites[site_id] = {
                        "id": site_id,
                        "status": "error",
                        "error": str(e)
                    }
        
        return sites

def get_importance(site_id: str) -> Dict[str, Any]:
    """Get feature importance for the specified site."""
    manager = get_model_manager(site_id)
    return manager.get_importance()

def update_site_model(site_id: str, model, scaler, feature_names: List[str]) -> bool:
    """Update the model for the specified site."""
    manager = get_model_manager(site_id)
    return manager.save_model(model, scaler, feature_names)

def update_site_configuration(site_id: str, config: Dict[str, Any]) -> bool:
    """Update the configuration for the specified site."""
    manager = get_model_manager(site_id)
    return manager.update_configuration(config)

def get_system_stats() -> Dict[str, Any]:
    """Get system statistics including memory usage, uptime, etc."""
    stats = {
        "uptime_seconds": time.time() - STATS["start_time"],
        "total_sites": len(get_site_info()),
        "active_sites": len([s for s_id, s in get_site_info().items() if s.get("status") == "active"]),
        "total_models": len(MODEL_CACHE),
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "total_predictions": STATS["total_predictions"],
        "recent_predictions": STATS["recent_predictions"]
    }
    
    # Calculate average prediction time
    if STATS["prediction_times"]:
        avg_time = sum(STATS["prediction_times"]) / len(STATS["prediction_times"])
        stats["avg_prediction_time_ms"] = avg_time * 1000
    else:
        stats["avg_prediction_time_ms"] = 0
    
    return stats


# Initialize directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Log startup information
logger.info(f"Realtime Model Manager initialized")
logger.info(f"Model directory: {os.path.abspath(MODELS_DIR)}")
logger.info(f"Config directory: {os.path.abspath(CONFIG_DIR)}")
logger.info(f"Max cached models: {MAX_CACHED_MODELS}") 