"""
Prediction module for Alzheimer's disease classification
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
import time
import logging
from pathlib import Path

from .cnn_model import AlzheimerCNN

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result dataclass"""
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PredictionResult':
        """Create from dictionary"""
        return cls(**data)


class Predictor:
    """
    Handles model prediction for single and batch images
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (optional)
        """
        self.model = None
        self.class_names = None
        self.input_shape = None
        self.model_path = model_path
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from file
        
        Args:
            model_path: Path to model file
        """
        try:
            # Load using AlzheimerCNN
            cnn_model = AlzheimerCNN()
            cnn_model.load(model_path)
            
            self.model = cnn_model.model
            self.class_names = cnn_model.class_names
            self.input_shape = cnn_model.input_shape
            self.model_path = model_path
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self,
                        image: Union[str, Path, bytes, np.ndarray, Image.Image],
                        target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image: Input image (file path, bytes, array, or PIL Image)
            target_size: Target size (defaults to model input shape)
            
        Returns:
            Preprocessed image array
        """
        if target_size is None:
            if self.input_shape:
                target_size = self.input_shape[:2]
            else:
                target_size = (224, 224)
        
        # Load image based on input type
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        return img_array
    
    def predict(self,
               image: Union[str, Path, bytes, np.ndarray, Image.Image],
               return_all_probs: bool = True) -> PredictionResult:
        """
        Predict single image
        
        Args:
            image: Input image
            return_all_probs: Whether to return all class probabilities
            
        Returns:
            PredictionResult object
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image)
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            
            # Get results
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx] * 100)
            
            # Get all probabilities
            all_probs = {}
            if return_all_probs and self.class_names:
                for i, class_name in enumerate(self.class_names):
                    # FIXED: Added closing bracket
                    all_probs[class_name] = float(predictions[0][i] * 100)
            
            # Sort by confidence
            all_probs = dict(sorted(
                all_probs.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            processing_time = time.time() - start_time
            
            result = PredictionResult(
                predicted_class=self.class_names[predicted_idx] if self.class_names else str(predicted_idx),
                predicted_class_idx=int(predicted_idx),
                confidence=confidence,
                all_probabilities=all_probs,
                processing_time=processing_time
            )
            
            logger.info(f"Prediction completed in {processing_time:.3f}s: {result.predicted_class} ({result.confidence:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                predicted_class="",
                predicted_class_idx=-1,
                confidence=0,
                all_probabilities={},
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def predict_batch(self,
                     images: List[Union[str, Path, bytes, np.ndarray, Image.Image]],
                     batch_size: int = 32) -> List[PredictionResult]:
        """
        Predict multiple images
        
        Args:
            images: List of input images
            batch_size: Batch size for prediction
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = []
            
            for image in batch:
                result = self.predict(image)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def predict_from_bytes(self, image_bytes: bytes) -> PredictionResult:
        """
        Predict from image bytes
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            PredictionResult
        """
        return self.predict(image_bytes)
    
    def predict_from_base64(self, base64_string: str) -> PredictionResult:
        """
        Predict from base64 encoded image
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            PredictionResult
        """
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        return self.predict(image_bytes)