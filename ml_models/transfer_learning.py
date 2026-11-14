"""
Transfer Learning models for Alzheimer's disease classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2, DenseNet121, DenseNet169,
    MobileNet, MobileNetV2, EfficientNetB0, EfficientNetB7,
    Xception, NASNetMobile
)
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
import logging

from .cnn_model import AlzheimerCNN

logger = logging.getLogger(__name__)

class TransferLearningModel(AlzheimerCNN):
    """
    Transfer Learning models wrapper
    """
    
    AVAILABLE_MODELS = {
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'inceptionv3': InceptionV3,
        'inceptionresnetv2': InceptionResNetV2,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'mobilenet': MobileNet,
        'mobilenetv2': MobileNetV2,
        'efficientnetb0': EfficientNetB0,
        'efficientnetb7': EfficientNetB7,
        'xception': Xception,
        'nasnetmobile': NASNetMobile
    }
    
    def __init__(self,
                 base_model_name: str = 'vgg16',
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 4,
                 class_names: List[str] = None):
        """
        Initialize transfer learning model
        
        Args:
            base_model_name: Name of base model
            input_shape: Input image shape
            num_classes: Number of classes
            class_names: List of class names
        """
        super().__init__(input_shape, num_classes, class_names)
        self.base_model_name = base_model_name.lower()
        self.base_model = None
        
        if self.base_model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {base_model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
    
    def build_model(self,
                   trainable_base: bool = False,
                   pooling: str = 'avg',
                   dropout_rate: float = 0.5,
                   include_top: bool = False) -> tf.keras.Model:
        """
        Build transfer learning model
        
        Args:
            trainable_base: Whether to make base model trainable
            pooling: Pooling type ('avg', 'max', None)
            dropout_rate: Dropout rate for custom layers
            include_top: Whether to include base model's top layers
            
        Returns:
            Keras model
        """
        # Get base model class
        base_class = self.AVAILABLE_MODELS[self.base_model_name]
        
        # Load base model
        self.base_model = base_class(
            weights='imagenet',
            include_top=include_top,
            input_shape=self.input_shape,
            pooling=pooling
        )
        
        # Freeze base model layers
        self.base_model.trainable = trainable_base
        
        # Build custom top layers
        model = models.Sequential()
        model.add(self.base_model)
        
        # Add custom layers
        if not include_top:
            if pooling is None:
                model.add(layers.GlobalAveragePooling2D())
            
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate * 0.6))
            
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return model
    
    def freeze_base_layers(self, freeze: bool = True):
        """Freeze or unfreeze base model layers"""
        if self.base_model:
            self.base_model.trainable = not freeze
    
    def unfreeze_top_layers(self, num_layers: int = 10):
        """Unfreeze top layers of base model for fine-tuning"""
        if self.base_model:
            # Freeze all layers first
            self.base_model.trainable = False
            
            # Unfreeze last num_layers
            for layer in self.base_model.layers[-num_layers:]:
                layer.trainable = True
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.base_model is None:
            return {}
        
        return {
            'base_model': self.base_model_name,
            'total_layers': len(self.base_model.layers),
            'trainable_params': self.base_model.trainable_weights,
            'non_trainable_params': self.base_model.non_trainable_weights,
            'input_shape': self.input_shape,
            'output_shape': self.model.output_shape if self.model else None
        }

class FineTuner:
    """
    Fine-tuning manager for transfer learning models
    """
    
    def __init__(self, model: TransferLearningModel):
        """
        Initialize fine-tuner
        
        Args:
            model: TransferLearningModel instance
        """
        self.model = model
        self.initial_learning_rate = None
        self.fine_tuning_learning_rate = None
    
    def prepare_for_fine_tuning(self,
                                initial_epochs: int,
                                fine_tune_epochs: int,
                                initial_lr: float = 0.001,
                                fine_tune_lr: float = 1e-5,
                                unfreeze_layers: int = 10):
        """
        Prepare model for fine-tuning
        
        Args:
            initial_epochs: Initial training epochs
            fine_tune_epochs: Fine-tuning epochs
            initial_lr: Initial learning rate
            fine_tune_lr: Fine-tuning learning rate
            unfreeze_layers: Number of top layers to unfreeze
        """
        self.initial_learning_rate = initial_lr
        self.fine_tuning_learning_rate = fine_tune_lr
        
        # Freeze base layers initially
        self.model.freeze_base_layers(True)
        
        # Compile for initial training
        self.model.compile_model(learning_rate=initial_lr)
    
    def fine_tune(self,
                 train_generator,
                 val_generator,
                 model_save_path: str,
                 unfreeze_layers: int = 10):
        """
        Perform fine-tuning
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            model_save_path: Path to save model
            unfreeze_layers: Number of top layers to unfreeze
            
        Returns:
            Training history
        """
        # Unfreeze top layers
        self.model.unfreeze_top_layers(unfreeze_layers)
        
        # Recompile with lower learning rate
        self.model.compile_model(learning_rate=self.fine_tuning_learning_rate)
        
        # Get callbacks
        callbacks = self.model.get_callbacks(
            checkpoint_path=model_save_path.replace('.h5', '_finetuned.h5'),
            early_stopping_patience=10,
            reduce_lr_patience=3
        )
        
        # Train
        history = self.model.model.fit(
            train_generator,
            epochs=len(callbacks) if callbacks else 50,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history

class ModelEnsemble:
    """
    Ensemble of multiple transfer learning models
    """
    
    def __init__(self, model_configs: List[Dict]):
        """
        Initialize ensemble
        
        Args:
            model_configs: List of model configurations
        """
        self.models = []
        self.weights = []
        
        for config in model_configs:
            model = TransferLearningModel(
                base_model_name=config['base_model'],
                input_shape=config.get('input_shape', (224, 224, 3)),
                num_classes=config.get('num_classes', 4)
            )
            model.build_model()
            
            if 'weights_path' in config:
                model.load(config['weights_path'])
            
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction
        
        Args:
            image: Input image
            
        Returns:
            Averaged predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model.model.predict(image[np.newaxis, ...], verbose=0)
            predictions.append(pred[0])
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights)
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        return weighted_pred
    
    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Batch prediction"""
        predictions = []
        
        for model in self.models:
            pred = model.model.predict(images, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights)
        
        # Reshape weights for broadcasting
        weights = weights.reshape(-1, 1, 1)
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        
        return weighted_pred

class ModelConfigurator:
    """
    Helper class for configuring transfer learning models
    """
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict:
        """Get model configuration"""
        configs = {
            'vgg16': {
                'input_size': 224,
                'layers': 16,
                'parameters': 138,
                'depth': 'medium',
                'best_for': 'general purpose'
            },
            'resnet50': {
                'input_size': 224,
                'layers': 50,
                'parameters': 25.6,
                'depth': 'deep',
                'best_for': 'balanced performance'
            },
            'inceptionv3': {
                'input_size': 299,
                'layers': 48,
                'parameters': 23.9,
                'depth': 'deep',
                'best_for': 'efficient computation'
            },
            'densenet121': {
                'input_size': 224,
                'layers': 121,
                'parameters': 8.1,
                'depth': 'very deep',
                'best_for': 'parameter efficiency'
            },
            'efficientnetb0': {
                'input_size': 224,
                'layers': 237,
                'parameters': 5.3,
                'depth': 'scalable',
                'best_for': 'state-of-the-art efficiency'
            }
        }
        
        return configs.get(model_name.lower(), {})
    
    @staticmethod
    def suggest_model(dataset_size: int, computational_budget: str = 'medium') -> str:
        """
        Suggest appropriate model based on dataset size and computational budget
        
        Args:
            dataset_size: Number of training samples
            computational_budget: 'low', 'medium', or 'high'
            
        Returns:
            Suggested model name
        """
        if dataset_size < 1000:
            if computational_budget == 'low':
                return 'mobilenet'
            elif computational_budget == 'medium':
                return 'vgg16'
            else:
                return 'resnet50'
        elif dataset_size < 5000:
            if computational_budget == 'low':
                return 'mobilenetv2'
            elif computational_budget == 'medium':
                return 'resnet50'
            else:
                return 'inceptionv3'
        else:
            if computational_budget == 'low':
                return 'efficientnetb0'
            elif computational_budget == 'medium':
                return 'densenet121'
            else:
                return 'inceptionresnetv2'
    
    @staticmethod
    def get_input_size(model_name: str) -> int:
        """Get required input size for model"""
        sizes = {
            'vgg16': 224,
            'vgg19': 224,
            'resnet50': 224,
            'resnet101': 224,
            'resnet152': 224,
            'inceptionv3': 299,
            'inceptionresnetv2': 299,
            'densenet121': 224,
            'densenet169': 224,
            'mobilenet': 224,
            'mobilenetv2': 224,
            'efficientnetb0': 224,
            'efficientnetb7': 600,
            'xception': 299,
            'nasnetmobile': 224
        }
        
        return sizes.get(model_name.lower(), 224)