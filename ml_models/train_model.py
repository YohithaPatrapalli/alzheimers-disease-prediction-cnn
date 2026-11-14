"""
Model training module for Alzheimer's disease classification
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from .cnn_model import AlzheimerCNN

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    optimizer: str = 'adam'
    loss: str = 'categorical_crossentropy'
    augmentation: bool = True
    class_weights: Optional[Dict] = None
    mixed_precision: bool = False
    num_workers: int = 4
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingConfig':
        """Create from dictionary"""
        return cls(**data)

class ModelTrainer:
    """
    Handles model training with various configurations
    """
    
    def __init__(self,
                 model: AlzheimerCNN,
                 config: TrainingConfig = None):
        """
        Initialize trainer
        
        Args:
            model: AlzheimerCNN model instance
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()
        self.history = None
        self.training_time = None
        self.best_epoch = None
        self.train_generator = None
        self.val_generator = None
        
    def prepare_data(self,
                    train_dir: str,
                    val_dir: str,
                    target_size: Tuple[int, int] = (224, 224)) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Prepare data generators with augmentation
        
        Args:
            train_dir: Path to training data directory
            val_dir: Path to validation data directory
            target_size: Target image size
            
        Returns:
            Tuple of train and validation generators
        """
        train_classes = sorted(
            name for name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, name))
        )
        val_classes = sorted(
            name for name in os.listdir(val_dir)
            if os.path.isdir(os.path.join(val_dir, name))
        )
        if not train_classes or train_classes != val_classes:
            raise ValueError(
                "Training and validation directories must contain the same class folders."
            )
        if len(train_classes) != self.model.num_classes:
            raise ValueError(
                f"Found {len(train_classes)} class folders, but the model expects "
                f"{self.model.num_classes} output classes."
            )

        self.model.class_names = train_classes

        # Training data generator with augmentation
        if self.config.augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2],
                channel_shift_range=0.1
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=True,
            classes=train_classes
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=train_classes
        )
        
        for split_name, generator in (
            ('training', self.train_generator),
            ('validation', self.val_generator),
        ):
            present_classes = set(generator.classes)
            missing_classes = [
                class_name for class_name, index in generator.class_indices.items()
                if index not in present_classes
            ]
            if missing_classes:
                raise ValueError(
                    f"{split_name.capitalize()} data has no images for classes: "
                    f"{', '.join(missing_classes)}."
                )
        
        return self.train_generator, self.val_generator
    
    def train(self,
             train_dir: str,
             val_dir: str,
             model_save_path: str,
             callbacks: List = None) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            model_save_path: Path to save the model
            callbacks: Additional callbacks
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        self.prepare_data(train_dir, val_dir)
        
        # Compile model if not compiled
        if not self.model.compiled:
            self.model.compile_model(
                optimizer=self.config.optimizer,
                learning_rate=self.config.learning_rate,
                loss=self.config.loss
            )
        
        # Get callbacks
        if callbacks is None:
            callbacks = self.model.get_callbacks(
                checkpoint_path=model_save_path,
                early_stopping_patience=self.config.early_stopping_patience,
                reduce_lr_patience=self.config.reduce_lr_patience
            )
        
        # Enable mixed precision if configured
        if self.config.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Train model
        start_time = time.time()
        
        self.history = self.model.model.fit(
            self.train_generator,
            epochs=self.config.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.config.class_weights,
            workers=self.config.num_workers,
            use_multiprocessing=True if self.config.num_workers > 1 else False
        )
        
        self.training_time = time.time() - start_time
        
        # Find best epoch
        val_acc = self.history.history.get('val_accuracy', [])
        if val_acc:
            self.best_epoch = np.argmax(val_acc) + 1
        
        # Save training config
        self._save_training_config(model_save_path)
        
        return self.history.history
    
    def train_from_generators(self,
                             train_generator: ImageDataGenerator,
                             val_generator: ImageDataGenerator,
                             model_save_path: str,
                             callbacks: List = None) -> Dict[str, Any]:
        """
        Train using existing data generators
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            model_save_path: Path to save model
            callbacks: Additional callbacks
            
        Returns:
            Training history
        """
        self.train_generator = train_generator
        self.val_generator = val_generator
        
        # Compile model if needed
        if not self.model.compiled:
            self.model.compile_model(
                optimizer=self.config.optimizer,
                learning_rate=self.config.learning_rate
            )
        
        # Get callbacks
        if callbacks is None:
            callbacks = self.model.get_callbacks(model_save_path)
        
        # Train
        start_time = time.time()
        
        self.history = self.model.model.fit(
            self.train_generator,
            epochs=self.config.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.config.class_weights
        )
        
        self.training_time = time.time() - start_time
        
        return self.history.history
    
    def resume_training(self,
                       train_dir: str,
                       val_dir: str,
                       model_path: str,
                       additional_epochs: int = 20) -> Dict[str, Any]:
        """
        Resume training from saved model
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            model_path: Path to saved model
            additional_epochs: Additional epochs to train
            
        Returns:
            Training history
        """
        # Load model
        self.model.load(model_path)
        
        # Update config
        self.config.epochs = additional_epochs
        
        # Resume training
        return self.train(train_dir, val_dir, model_path)
    
    def get_learning_curves(self) -> Dict[str, List]:
        """Get learning curves data"""
        if self.history is None:
            return {}
        
        return {
            'accuracy': self.history.history.get('accuracy', []),
            'val_accuracy': self.history.history.get('val_accuracy', []),
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'precision': self.history.history.get('precision', []),
            'val_precision': self.history.history.get('val_precision', []),
            'recall': self.history.history.get('recall', []),
            'val_recall': self.history.history.get('val_recall', [])
        }
    
    def _save_training_config(self, model_path: str):
        """Save training configuration"""
        config_data = {
            'config': self.config.to_dict(),
            'training_time': self.training_time,
            'best_epoch': self.best_epoch,
            'final_metrics': {
                'accuracy': self.history.history.get('accuracy', [None])[-1],
                'val_accuracy': self.history.history.get('val_accuracy', [None])[-1],
                'loss': self.history.history.get('loss', [None])[-1],
                'val_loss': self.history.history.get('val_loss', [None])[-1]
            } if self.history else {},
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = model_path.replace('.h5', '_training_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        if self.history is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CrossValidator:
    """
    K-Fold Cross Validation for model evaluation
    """
    
    def __init__(self, model_class, config: TrainingConfig, n_folds: int = 5):
        self.model_class = model_class
        self.config = config
        self.n_folds = n_folds
        self.fold_histories = []
        self.fold_scores = []
        
    def validate(self, data_dir: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Perform k-fold cross validation
        
        Args:
            data_dir: Directory containing class subdirectories
            target_size: Target image size
        """
        from sklearn.model_selection import KFold
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Create data generator
        datagen = ImageDataGenerator(rescale=1./255)
        
        # Get all image paths and labels
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Prepare data for k-fold
        n_samples = generator.samples
        indices = np.arange(n_samples)
        labels = generator.classes
        
        # K-Fold
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, labels)):
            logger.info(f"Training fold {fold + 1}/{self.n_folds}")
            
            # Create model
            model = self.model_class(
                input_shape=(*target_size, 3),
                num_classes=generator.num_classes
            )
            model.build_model()
            
            # Create trainer
            trainer = ModelTrainer(model, self.config)
            
            # Train
            history = trainer.train(
                train_dir=data_dir,  # This needs proper split handling
                val_dir=data_dir,    # In practice, you'd need to create train/val splits
                model_save_path=f'model_fold_{fold}.h5'
            )
            
            self.fold_histories.append(history)
            
            # Evaluate
            val_acc = history.history.get('val_accuracy', [0])[-1]
            self.fold_scores.append(val_acc)
        
        return {
            'fold_scores': self.fold_scores,
            'mean_score': np.mean(self.fold_scores),
            'std_score': np.std(self.fold_scores)
        }