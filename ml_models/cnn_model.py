"""
Custom CNN Model for Alzheimer's Disease Classification
Supports 4-stage classification: Non-Demented, Very Mild, Mild, Moderate
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import logging
from tensorflow.keras.layers import InputLayer
from tensorflow.keras import mixed_precision

# PATCH 1: batch_shape fix
original_init = InputLayer.__init__

def new_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    original_init(self, *args, **kwargs)

InputLayer.__init__ = new_init

    

logger = logging.getLogger(__name__)

class AlzheimerCNN:
    """
    Custom CNN architecture for Alzheimer's disease classification
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 4,
                 class_names: List[str] = None):
        """
        Initialize CNN model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            class_names: List of class names
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = class_names or [
            'Non-Demented',
            'Very Mild Demented',
            'Mild Demented',
            'Moderate Demented'
        ]
        self.model = None
        self.history = None
        self.compiled = False
        
    def build_model(self) -> tf.keras.Model:
        """
        Build custom CNN architecture with multiple convolutional blocks
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation (only during training)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fifth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_lightweight_model(self) -> tf.keras.Model:
        """
        Build lightweight CNN for faster inference
        
        Returns:
            Lightweight Keras model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self,
                     optimizer: str = 'adam',
                     learning_rate: float = 0.001,
                     loss: str = 'categorical_crossentropy',
                     metrics: List[str] = None) -> None:
        """
        Compile the model with specified parameters
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate for optimizer
            loss: Loss function
            metrics: List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if metrics is None:
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        
        # Configure optimizer
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.compiled = True
        logger.info(f"Model compiled with {optimizer}, learning rate: {learning_rate}")
    
    def get_callbacks(self, 
                     checkpoint_path: str,
                     early_stopping_patience: int = 15,
                     reduce_lr_patience: int = 5) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks
        
        Args:
            checkpoint_path: Path to save model checkpoints
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger
            tf.keras.callbacks.CSVLogger(
                filename=checkpoint_path.replace('.h5', '_training.log'),
                separator=',',
                append=False
            ),
            
            # Learning rate scheduler
            tf.keras.callbacks.LearningRateScheduler(
                self._lr_scheduler,
                verbose=0
            )
        ]
        
        return callbacks
    
    def _lr_scheduler(self, epoch: int) -> float:
        """
        Learning rate scheduler
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Adjusted learning rate
        """
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 20.0
        
        lr = initial_lr * (drop ** np.floor((epoch) / epochs_drop))
        return lr
    
    def summary(self) -> str:
        """Get model summary as string"""
        if self.model is None:
            return "Model not built yet"
        
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    
    def save(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(filepath)
        
        # Save class names and config
        config = {
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        config_path = filepath.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {filepath}")
    


    def load(self, filepath: str) -> None:
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        self.model = tf.keras.models.load_model(
        filepath,
        compile=False,
        custom_objects={
            "DTypePolicy": mixed_precision.Policy
        })
        
        # Load config if exists
        config_path = filepath.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.class_names = config.get('class_names', self.class_names)
                self.input_shape = tuple(config.get('input_shape', self.input_shape))
                self.num_classes = config.get('num_classes', self.num_classes)
        
        self.compiled = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_layer_outputs(self, image: np.ndarray, layer_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get outputs of intermediate layers for visualization
        
        Args:
            image: Input image (preprocessed)
            layer_names: List of layer names to get outputs from
            
        Returns:
            Dictionary of layer names to outputs
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if layer_names is None:
            # Get all convolutional layers
            layer_names = [
                layer.name for layer in self.model.layers 
                if isinstance(layer, layers.Conv2D)
            ]
        
        # Create intermediate model
        intermediate_outputs = [self.model.get_layer(name).output for name in layer_names]
        intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=intermediate_outputs)
        
        # Get outputs
        outputs = intermediate_model.predict(image[np.newaxis, ...])
        
        return dict(zip(layer_names, outputs))
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and non-trainable parameters"""
        if self.model is None:
            return {'total': 0, 'trainable': 0, 'non_trainable': 0}
        
        total = self.model.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': non_trainable
        }

class ModelFactory:
    """
    Factory class for creating different model architectures
    """
    
    @staticmethod
    def create_model(model_type: str,
                    input_shape: Tuple[int, int, int] = (224, 224, 3),
                    num_classes: int = 4,
                    **kwargs) -> AlzheimerCNN:
        """
        Create model by type
        
        Args:
            model_type: Type of model ('cnn', 'lightweight', 'deep', 'residual')
            input_shape: Input image shape
            num_classes: Number of classes
            **kwargs: Additional arguments
            
        Returns:
            AlzheimerCNN instance
        """
        model = AlzheimerCNN(input_shape=input_shape, num_classes=num_classes)
        
        if model_type == 'cnn':
            model.build_model()
        elif model_type == 'lightweight':
            model.build_lightweight_model()
        elif model_type == 'deep':
            model = ModelFactory._build_deep_cnn(input_shape, num_classes)
        elif model_type == 'residual':
            model = ModelFactory._build_residual_cnn(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    @staticmethod
    def _build_deep_cnn(input_shape, num_classes):
        """Build deeper CNN architecture"""
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            
            # Block 4
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.GlobalAveragePooling2D(),
            
            # Classifier
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        cnn_model = AlzheimerCNN(input_shape=input_shape, num_classes=num_classes)
        cnn_model.model = model
        return cnn_model
    
    @staticmethod
    def _build_residual_cnn(input_shape, num_classes):
        """Build CNN with residual connections"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = ModelFactory._residual_block(x, 64, 3)
        x = ModelFactory._residual_block(x, 128, 4, stride=2)
        x = ModelFactory._residual_block(x, 256, 6, stride=2)
        x = ModelFactory._residual_block(x, 512, 3, stride=2)
        
        # Classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        cnn_model = AlzheimerCNN(input_shape=input_shape, num_classes=num_classes)
        cnn_model.model = model
        return cnn_model
    
    @staticmethod
    def _residual_block(x, filters, repetitions, stride=1):
        """Create residual block"""
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, 1, strides=stride, padding='valid')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Middle convolutions
        for _ in range(repetitions - 2):
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        
        # Last convolution
        x = layers.Conv2D(filters * 4, 1, padding='valid')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters * 4:
            shortcut = layers.Conv2D(filters * 4, 1, strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x