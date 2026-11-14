import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from django.conf import settings
import os
import json
import pickle
import joblib
from PIL import Image
import cv2
from io import BytesIO
import base64

class AlzheimerCNN:
    """
    CNN Model for Alzheimer's Disease Classification
    Supports 4 stages: Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
        self.history = None
        
    def build_model(self, model_type='cnn'):
        """
        Build CNN model based on architecture type
        Args:
            model_type: 'cnn', 'vgg16', 'resnet50', 'inceptionv3'
        """
        if model_type == 'cnn':
            self.model = self._build_custom_cnn()
        elif model_type == 'vgg16':
            self.model = self._build_transfer_learning(VGG16, 'vgg16')
        elif model_type == 'resnet50':
            self.model = self._build_transfer_learning(ResNet50, 'resnet50')
        elif model_type == 'inceptionv3':
            self.model = self._build_transfer_learning(InceptionV3, 'inceptionv3')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model
    
    def _build_custom_cnn(self):
        """Build custom CNN architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
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
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_transfer_learning(self, base_model_class, model_name):
        """Build transfer learning model"""
        # Load pre-trained model
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with optimizer and metrics"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def get_callbacks(self, model_path):
        """Get training callbacks"""
        callbacks = [
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def get_data_generators(self, train_dir, val_dir, test_dir, batch_size=32):
        """Create data generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # Validation and test data generator (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            classes=self.class_names
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=self.class_names
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            classes=self.class_names
        )
        
        # Save class indices
        self.class_indices = train_generator.class_indices
        
        return train_generator, val_generator, test_generator
    
    def train(self, train_generator, val_generator, epochs=50, callbacks=None):
        """Train the model"""
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)
        
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate model on test data"""
        results = self.model.evaluate(test_generator)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def predict(self, image_path):
        """
        Predict single image
        Args:
            image_path: path to image file or image array
        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=self.input_shape[:2]
            )
        else:
            # Assume it's a PIL Image
            img = image_path.resize(self.input_shape[:2])
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(self.class_names):
            class_probabilities[class_name] = float(predictions[0][i] * 100)
        
        # Sort probabilities
        sorted_probs = dict(sorted(
            class_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        result = {
            'predicted_class': self.class_names[predicted_class_idx],
            'predicted_class_idx': int(predicted_class_idx),
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'success': True
        }
        
        return result
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'success': False,
                    'image_path': image_path
                })
        
        return results
    
    def save_model(self, filepath):
        """Save model to file"""
        # Save model architecture and weights
        self.model.save(filepath)
        
        # Save class indices
        if hasattr(self, 'class_indices'):
            indices_path = filepath.replace('.h5', '_indices.json')
            with open(indices_path, 'w') as f:
                json.dump(self.class_indices, f)
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = tf.keras.models.load_model(filepath)
        
        # Load class indices if available
        indices_path = filepath.replace('.h5', '_indices.json')
        if os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                self.class_indices = json.load(f)
        
        return self.model
    
    def get_model_summary(self):
        """Get model summary as string"""
        stringio = tf.keras.utils.io_utils.file_io.FileIO()
        self.model.summary(print_fn=lambda x: stringio.write(x + '\n'))
        return stringio.getvalue()

class ImagePreprocessor:
    """Image preprocessing utilities"""
    
    @staticmethod
    def preprocess_image(image_file, target_size=(224, 224)):
        """Preprocess image for prediction"""
        # Read image
        if isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            img = Image.open(image_file)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
    
    @staticmethod
    def normalize_image(img_array):
        """Normalize image array to [0,1]"""
        return img_array / 255.0
    
    @staticmethod
    def apply_clahe(image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Apply CLAHE to each channel
        if len(img_array.shape) == 3:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            for i in range(3):
                img_array[:,:,i] = clahe.apply(img_array[:,:,i])
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def image_to_base64(image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, classification_report
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    @staticmethod
    def calculate_per_class_metrics(y_true, y_pred, class_names):
        """Calculate per-class metrics"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        
        per_class = {}
        for i, class_name in enumerate(class_names):
            per_class[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return per_class