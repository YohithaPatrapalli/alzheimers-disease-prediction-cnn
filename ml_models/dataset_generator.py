"""
Dataset Generator for Alzheimer's Disease Prediction System
Generates synthetic MRI-like images for training and testing
"""

import numpy as np
import os
import random
import shutil
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from pathlib import Path
import json
import hashlib
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union
import time

# Add this import - OpenCV for image processing
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlzheimerDatasetGenerator:
    """
    Synthetic dataset generator for Alzheimer's disease classification
    Generates MRI-like images with controlled characteristics for each stage
    """
    
    def __init__(self, output_dir: str, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize dataset generator
        
        Args:
            output_dir: Directory to save generated dataset
            image_size: Size of generated images
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.classes = {
            'non_demented': 0,
            'very_mild': 1,
            'mild': 2,
            'moderate': 3
        }
        
        # Class-specific characteristics
        self.class_features = {
            'non_demented': {
                'brain_volume': (0.95, 1.05),
                'ventricle_size': (0.5, 0.7),
                'hippocampus_size': (0.9, 1.0),
                'texture_roughness': (0.3, 0.5),
                'atrophy_level': (0, 0.1),
                'white_matter_intensity': (0.7, 0.9),
                'gray_matter_intensity': (0.5, 0.7)
            },
            'very_mild': {
                'brain_volume': (0.9, 0.98),
                'ventricle_size': (0.65, 0.8),
                'hippocampus_size': (0.8, 0.9),
                'texture_roughness': (0.45, 0.65),
                'atrophy_level': (0.1, 0.25),
                'white_matter_intensity': (0.65, 0.85),
                'gray_matter_intensity': (0.45, 0.65)
            },
            'mild': {
                'brain_volume': (0.8, 0.9),
                'ventricle_size': (0.75, 0.9),
                'hippocampus_size': (0.65, 0.8),
                'texture_roughness': (0.6, 0.8),
                'atrophy_level': (0.25, 0.45),
                'white_matter_intensity': (0.55, 0.75),
                'gray_matter_intensity': (0.35, 0.55)
            },
            'moderate': {
                'brain_volume': (0.7, 0.82),
                'ventricle_size': (0.85, 1.0),
                'hippocampus_size': (0.45, 0.65),
                'texture_roughness': (0.75, 0.95),
                'atrophy_level': (0.45, 0.65),
                'white_matter_intensity': (0.45, 0.65),
                'gray_matter_intensity': (0.25, 0.45)
            }
        }
        
        self.create_directories()
    
    def create_directories(self):
        """Create output directories for each class"""
        for class_name in self.classes.keys():
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories in {self.output_dir}")
    
    def generate_brain_mask(self, size: Tuple[int, int], 
                           brain_volume: float,
                           atrophy_level: float) -> np.ndarray:
        """
        Generate a brain-shaped mask
        
        Args:
            size: Image size
            brain_volume: Relative size of brain region
            atrophy_level: Level of brain atrophy
            
        Returns:
            Binary mask of brain region
        """
        h, w = size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create elliptical shape for brain
        center_x = w // 2
        center_y = h // 2
        radius_x = int(w * brain_volume * 0.35 * (1 - atrophy_level * 0.3))
        radius_y = int(h * brain_volume * 0.4 * (1 - atrophy_level * 0.2))
        
        # Draw brain shape with irregularities
        for y in range(h):
            for x in range(w):
                # Ellipse equation with noise
                dx = (x - center_x) / radius_x
                dy = (y - center_y) / radius_y
                
                # Add some irregularity to simulate brain shape
                noise = np.random.randn() * 0.05
                if dx**2 + dy**2 <= 1 + noise:
                    mask[y, x] = 255
        
        # Add some holes to simulate brain structures
        for _ in range(int(10 * (1 + atrophy_level))):
            cx = center_x + np.random.randint(-radius_x//2, radius_x//2)
            cy = center_y + np.random.randint(-radius_y//2, radius_y//2)
            r = np.random.randint(3, 8)
            cv2.circle(mask, (cx, cy), r, 0, -1)
        
        return mask
    
    def generate_texture(self, size: Tuple[int, int], 
                        roughness: float,
                        intensity_range: Tuple[float, float]) -> np.ndarray:
        """
        Generate realistic brain tissue texture
        
        Args:
            size: Image size
            roughness: Texture roughness
            intensity_range: Intensity range (min, max)
            
        Returns:
            Texture array
        """
        h, w = size
        
        # Generate base noise
        base_noise = np.random.randn(h, w)
        
        # Create fractal noise for realistic texture
        for octave in range(3):
            scale = 2 ** octave
            noise = np.random.randn(h // scale, w // scale)
            noise = np.kron(noise, np.ones((scale, scale)))[:h, :w]
            base_noise += noise * (0.5 ** octave)
        
        # Normalize and apply intensity range
        base_noise = (base_noise - base_noise.min()) / (base_noise.max() - base_noise.min())
        base_noise = base_noise * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
        
        # Apply roughness filter
        from scipy.ndimage import gaussian_filter
        texture = gaussian_filter(base_noise, sigma=roughness * 3)
        
        return texture.astype(np.float32)
    
    def generate_ventricles(self, size: Tuple[int, int],
                           ventricle_size: float,
                           atrophy_level: float) -> np.ndarray:
        """
        Generate ventricle regions (enlarged in Alzheimer's)
        
        Args:
            size: Image size
            ventricle_size: Relative size of ventricles
            atrophy_level: Atrophy level affecting ventricle shape
            
        Returns:
            Ventricle mask
        """
        h, w = size
        ventricle_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Main ventricle (lateral)
        center_x = w // 2
        center_y = h // 2
        radius_x = int(w * ventricle_size * 0.15)
        radius_y = int(h * ventricle_size * 0.1)
        
        # Draw butterfly-shaped ventricles
        for y in range(h):
            for x in range(w):
                # Left ventricle
                dx_left = (x - (center_x - radius_x//2)) / radius_x
                dy = (y - center_y) / radius_y
                if dx_left**2 + dy**2 <= 1:
                    ventricle_mask[y, x] = 200
                
                # Right ventricle
                dx_right = (x - (center_x + radius_x//2)) / radius_x
                if dx_right**2 + dy**2 <= 1:
                    ventricle_mask[y, x] = 200
        
        # Add irregularity based on atrophy
        if atrophy_level > 0.3:
            for _ in range(int(atrophy_level * 10)):
                cx = center_x + np.random.randint(-radius_x, radius_x)
                cy = center_y + np.random.randint(-radius_y, radius_y)
                r = np.random.randint(2, 5)
                cv2.circle(ventricle_mask, (cx, cy), r, 180, -1)
        
        return ventricle_mask
    
    def generate_hippocampus(self, size: Tuple[int, int],
                            hippocampus_size: float,
                            atrophy_level: float) -> np.ndarray:
        """
        Generate hippocampus regions (atrophied in Alzheimer's)
        
        Args:
            size: Image size
            hippocampus_size: Relative size of hippocampus
            atrophy_level: Atrophy level
            
        Returns:
            Hippocampus mask
        """
        h, w = size
        hippo_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Hippocampus locations (temporal lobe)
        center_x_left = w // 3
        center_x_right = 2 * w // 3
        center_y = 2 * h // 3
        
        size_factor = hippocampus_size * (1 - atrophy_level * 0.5)
        radius_x = int(w * 0.08 * size_factor)
        radius_y = int(h * 0.05 * size_factor)
        
        # Draw left hippocampus
        cv2.ellipse(hippo_mask, (center_x_left, center_y), 
                   (radius_x, radius_y), 0, 0, 360, 200, -1)
        
        # Draw right hippocampus
        cv2.ellipse(hippo_mask, (center_x_right, center_y), 
                   (radius_x, radius_y), 0, 0, 360, 200, -1)
        
        # Add atrophy effects
        if atrophy_level > 0.2:
            # Shrink and deform hippocampus
            erosion_size = int(atrophy_level * 3)
            if erosion_size > 0:
                kernel = np.ones((erosion_size, erosion_size), np.uint8)
                hippo_mask = cv2.erode(hippo_mask, kernel, iterations=1)
        
        return hippo_mask
    
    def generate_image(self, class_name: str, 
                      subject_id: int,
                      seed: Optional[int] = None) -> Image.Image:
        """
        Generate a single synthetic MRI image
        
        Args:
            class_name: Class name ('non_demented', 'very_mild', 'mild', 'moderate')
            subject_id: Subject identifier
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        features = self.class_features[class_name]
        
        # Generate base components
        brain_volume = np.random.uniform(*features['brain_volume'])
        ventricle_size = np.random.uniform(*features['ventricle_size'])
        hippocampus_size = np.random.uniform(*features['hippocampus_size'])
        texture_roughness = np.random.uniform(*features['texture_roughness'])
        atrophy_level = np.random.uniform(*features['atrophy_level'])
        
        white_matter_intensity = np.random.uniform(*features['white_matter_intensity'])
        gray_matter_intensity = np.random.uniform(*features['gray_matter_intensity'])
        
        # Generate components
        brain_mask = self.generate_brain_mask(self.image_size, brain_volume, atrophy_level)
        brain_texture = self.generate_texture(self.image_size, texture_roughness, 
                                             (gray_matter_intensity, white_matter_intensity))
        
        ventricles = self.generate_ventricles(self.image_size, ventricle_size, atrophy_level)
        hippocampus = self.generate_hippocampus(self.image_size, hippocampus_size, atrophy_level)
        
        # Combine components
        image_array = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)
        
        # Apply brain texture to mask
        image_array = brain_texture * (brain_mask / 255.0)
        
        # Add ventricles (dark regions)
        image_array = np.where(ventricles > 0, image_array * 0.7, image_array)
        
        # Add hippocampus (slightly brighter)
        image_array = np.where(hippocampus > 0, image_array * 1.1, image_array)
        
        # Add noise
        noise_level = np.random.uniform(0.02, 0.08)
        image_array += np.random.randn(*self.image_size) * noise_level
        
        # Add subtle artifacts (simulating MRI acquisition)
        if np.random.random() < 0.3:
            # Add motion artifact
            motion_strength = np.random.uniform(0, 0.05)
            image_array = self.add_motion_artifact(image_array, motion_strength)
        
        if np.random.random() < 0.2:
            # Add intensity inhomogeneity
            image_array = self.add_intensity_inhomogeneity(image_array)
        
        # Normalize to 0-255 range
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
        image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(image_array, mode='L')
        
        # Apply subtle colorization for grayscale MRI look
        image = Image.merge('RGB', (image, image, image))
        
        # Add some noise to make it look more realistic
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(np.random.uniform(0.9, 1.1))
        
        return image
    
    def add_motion_artifact(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        Add simulated motion artifact to image
        
        Args:
            image: Input image array
            strength: Artifact strength
            
        Returns:
            Image with motion artifact
        """
        h, w = image.shape
        artifact = np.zeros_like(image)
        
        # Create ghosting effect
        shift = int(strength * 20)
        if shift > 0:
            artifact[:, shift:] = image[:, :-shift] * strength
            artifact[:, :shift] = image[:, :shift]
        
        return image + artifact
    
    def add_intensity_inhomogeneity(self, image: np.ndarray) -> np.ndarray:
        """
        Add intensity inhomogeneity (bias field) artifact
        
        Args:
            image: Input image array
            
        Returns:
            Image with bias field
        """
        h, w = image.shape
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create smooth bias field
        bias = 1 + 0.2 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        bias += 0.1 * np.cos(3 * np.pi * X) * np.sin(3 * np.pi * Y)
        
        return image * bias
    
    def generate_dataset(self, samples_per_class: Dict[str, int],
                        output_format: str = 'png',
                        num_workers: int = 4,
                        save_metadata: bool = True) -> Dict:
        """
        Generate full dataset
        
        Args:
            samples_per_class: Dictionary with class names and number of samples
            output_format: Image format ('png', 'jpg', 'tiff')
            num_workers: Number of parallel workers
            save_metadata: Whether to save metadata file
            
        Returns:
            Dictionary with generation statistics
        """
        logger.info("Starting dataset generation...")
        start_time = time.time()
        
        stats = {
            'total_images': 0,
            'per_class': {},
            'generation_time': 0
        }
        
        # Generate images for each class
        for class_name, num_samples in samples_per_class.items():
            if class_name not in self.classes:
                logger.warning(f"Unknown class: {class_name}, skipping")
                continue
            
            logger.info(f"Generating {num_samples} images for class: {class_name}")
            class_dir = self.output_dir / class_name
            
            # Use thread pool for parallel generation
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i in range(num_samples):
                    future = executor.submit(self.generate_image, class_name, i, i)
                    futures.append(future)
                
                for idx, future in enumerate(tqdm(as_completed(futures), total=num_samples, desc=class_name)):
                    try:
                        image = future.result()
                        image_path = class_dir / f"{class_name}_{idx:05d}.{output_format}"
                        image.save(image_path, quality=95)
                        stats['total_images'] += 1
                    except Exception as e:
                        logger.error(f"Error generating image {idx}: {e}")
            
            stats['per_class'][class_name] = num_samples
        
        # Generate metadata
        if save_metadata:
            metadata = self.generate_metadata(stats)
            metadata_path = self.output_dir / 'metadata' / 'dataset_info.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
        
        stats['generation_time'] = time.time() - start_time
        logger.info(f"Dataset generation completed in {stats['generation_time']:.2f} seconds")
        logger.info(f"Total images generated: {stats['total_images']}")
        
        return stats
    
    def generate_metadata(self, stats: Dict) -> Dict:
        """
        Generate metadata for the dataset
        
        Args:
            stats: Generation statistics
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'dataset_name': 'Synthetic_Alzheimer_MRI',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'image_size': self.image_size,
            'classes': self.classes,
            'samples_per_class': stats['per_class'],
            'total_samples': stats['total_images'],
            'class_features': self.class_features,
            'generation_parameters': {
                'image_size': self.image_size,
                'class_features': self.class_features
            }
        }
        
        return metadata
    
    def create_train_val_test_split(self, train_ratio: float = 0.7,
                                    val_ratio: float = 0.15,
                                    test_ratio: float = 0.15):
        """
        Create train/validation/test split from generated dataset
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        from sklearn.model_selection import train_test_split
        
        split_dir = self.output_dir / 'splits'
        split_dir.mkdir(exist_ok=True)
        
        for class_name in self.classes.keys():
            class_dir = self.output_dir / class_name
            images = list(class_dir.glob('*.*'))
            images = [str(img) for img in images]
            
            if not images:
                continue
            
            # Split into train and temp
            train_images, temp_images = train_test_split(
                images, test_size=(val_ratio + test_ratio), random_state=42
            )
            
            # Split temp into val and test
            val_images, test_images = train_test_split(
                temp_images, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
            )
            
            # Create split directories
            for split_name, split_images in [('train', train_images), 
                                            ('val', val_images), 
                                            ('test', test_images)]:
                split_class_dir = split_dir / split_name / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_images:
                    img = Image.open(img_path)
                    dest_path = split_class_dir / Path(img_path).name
                    img.save(dest_path)
        
        logger.info(f"Dataset split created at {split_dir}")
        
        # Save split info
        split_info = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'split_dir': str(split_dir)
        }
        
        with open(self.output_dir / 'metadata' / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)


class LargeScaleDatasetGenerator:
    """
    Large-scale dataset generator with memory-efficient batch processing
    """
    
    def __init__(self, output_dir: str, image_size: Tuple[int, int] = (224, 224)):
        self.generator = AlzheimerDatasetGenerator(output_dir, image_size)
        self.output_dir = Path(output_dir)
    
    def generate_large_dataset(self, 
                              samples_per_class: Dict[str, int],
                              batch_size: int = 1000,
                              output_format: str = 'png',
                              save_metadata: bool = True) -> Dict:
        """
        Generate large dataset in batches to manage memory
        
        Args:
            samples_per_class: Samples per class
            batch_size: Number of images to generate per batch
            output_format: Image format
            save_metadata: Save metadata flag
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_images': 0,
            'per_class': {},
            'batches': []
        }
        
        for class_name, num_samples in samples_per_class.items():
            logger.info(f"Generating {num_samples} images for {class_name}")
            class_dir = self.output_dir / class_name
            
            images_generated = 0
            batch_num = 0
            
            while images_generated < num_samples:
                batch_num += 1
                batch_count = min(batch_size, num_samples - images_generated)
                
                logger.info(f"Generating batch {batch_num} for {class_name} ({batch_count} images)")
                batch_start = time.time()
                
                # Generate batch
                batch_images = []
                for i in range(batch_count):
                    idx = images_generated + i
                    image = self.generator.generate_image(class_name, idx, idx)
                    batch_images.append(image)
                
                # Save batch
                for i, image in enumerate(batch_images):
                    idx = images_generated + i
                    image_path = class_dir / f"{class_name}_{idx:05d}.{output_format}"
                    image.save(image_path, quality=95)
                
                batch_time = time.time() - batch_start
                images_generated += batch_count
                
                stats['batches'].append({
                    'class': class_name,
                    'batch': batch_num,
                    'count': batch_count,
                    'time': batch_time
                })
                
                logger.info(f"Batch {batch_num} completed in {batch_time:.2f}s")
            
            stats['per_class'][class_name] = images_generated
            stats['total_images'] += images_generated
        
        if save_metadata:
            self._save_batch_metadata(stats)
        
        return stats
    
    def _save_batch_metadata(self, stats: Dict):
        """Save batch processing metadata"""
        metadata = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': stats['total_images'],
            'per_class': stats['per_class'],
            'batches': stats['batches']
        }
        
        with open(self.output_dir / 'metadata' / 'batch_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_with_progress_callback(self, 
                                        samples_per_class: Dict[str, int],
                                        callback: callable,
                                        batch_size: int = 1000):
        """
        Generate dataset with progress callback
        
        Args:
            samples_per_class: Samples per class
            callback: Progress callback function
            batch_size: Batch size
        """
        total_samples = sum(samples_per_class.values())
        processed = 0
        
        for class_name, num_samples in samples_per_class.items():
            class_dir = self.output_dir / class_name
            
            for i in range(num_samples):
                image = self.generator.generate_image(class_name, i, i)
                image_path = class_dir / f"{class_name}_{i:05d}.png"
                image.save(image_path, quality=95)
                
                processed += 1
                
                # Call progress callback
                if callback:
                    callback(processed, total_samples, class_name, i)


def generate_dataset_from_config(config_path: str, output_dir: str):
    """
    Generate dataset from configuration file
    
    Args:
        config_path: Path to JSON configuration file
        output_dir: Output directory
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    image_size = tuple(config.get('image_size', [224, 224]))
    samples_per_class = config.get('samples_per_class', {
        'non_demented': 1000,
        'very_mild': 1000,
        'mild': 1000,
        'moderate': 1000
    })
    
    generator = LargeScaleDatasetGenerator(output_dir, image_size)
    stats = generator.generate_large_dataset(samples_per_class, 
                                            batch_size=config.get('batch_size', 1000))
    
    # Create train/val/test split
    if config.get('create_split', True):
        split_ratios = config.get('split_ratios', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        split_generator = AlzheimerDatasetGenerator(output_dir, image_size)
        split_generator.create_train_val_test_split(**split_ratios)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic Alzheimer\'s MRI dataset')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for generated dataset')
    parser.add_argument('--samples', '-s', type=int, default=1000,
                       help='Number of samples per class')
    parser.add_argument('--size', type=int, default=224,
                       help='Image size (width/height)')
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration JSON file')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--format', '-f', type=str, default='png',
                       choices=['png', 'jpg', 'tiff'],
                       help='Output image format')
    
    args = parser.parse_args()
    
    if args.config:
        # Generate from config file
        stats = generate_dataset_from_config(args.config, args.output)
        print(f"Dataset generation completed: {stats['total_images']} images generated")
    else:
        # Generate with default parameters
        samples_per_class = {
            'non_demented': args.samples,
            'very_mild': args.samples,
            'mild': args.samples,
            'moderate': args.samples
        }
        
        generator = LargeScaleDatasetGenerator(args.output, (args.size, args.size))
        stats = generator.generate_large_dataset(samples_per_class, 
                                                batch_size=1000,
                                                output_format=args.format)
        
        print(f"\nDataset Generation Complete!")
        print(f"Total images generated: {stats['total_images']}")
        print(f"Output directory: {args.output}")
        print(f"Image size: {args.size}x{args.size}")
        print(f"Format: {args.format}")
        
        # Create split
        split_generator = AlzheimerDatasetGenerator(args.output, (args.size, args.size))
        split_generator.create_train_val_test_split()
        print(f"Train/Validation/Test split created")