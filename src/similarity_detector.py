"""Image similarity detection using various algorithms."""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import imagehash
import cv2

from config.settings import settings

logger = logging.getLogger(__name__)

class SimilarityDetector:
    """Handles image similarity detection using multiple algorithms."""
    
    def __init__(self):
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.supported_methods = ['phash', 'dhash', 'ahash', 'whash', 'histogram']
    
    def calculate_perceptual_hash(self, image_path: Path, method: str = 'phash') -> Optional[str]:
        """
        Calculate perceptual hash for an image.
        
        Args:
            image_path: Path to image file
            method: Hash method ('phash', 'dhash', 'ahash', 'whash')
            
        Returns:
            Hash string or None if failed
        """
        try:
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            image = Image.open(image_path)
            
            if method == 'phash':
                hash_obj = imagehash.phash(image)
            elif method == 'dhash':
                hash_obj = imagehash.dhash(image)
            elif method == 'ahash':
                hash_obj = imagehash.average_hash(image)
            elif method == 'whash':
                hash_obj = imagehash.whash(image)
            else:
                logger.error(f"Unsupported hash method: {method}")
                return None
            
            return str(hash_obj)
            
        except Exception as e:
            logger.error(f"Error calculating {method} for {image_path}: {str(e)}")
            return None
    
    def calculate_histogram_features(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Calculate color histogram features for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Histogram feature vector or None if failed
        """
        try:
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Read image with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Calculate histograms for each channel
            hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
            
            # Normalize histograms
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            
            # Combine histograms
            features = np.concatenate([hist_r, hist_g, hist_b])
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating histogram features for {image_path}: {str(e)}")
            return None
    
    def calculate_similarity_score(self, image1_path: Path, image2_path: Path, 
                                 method: str = 'phash') -> Optional[float]:
        """
        Calculate similarity score between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            method: Similarity method to use
            
        Returns:
            Similarity score (0.0 to 1.0) or None if failed
        """
        try:
            if method in ['phash', 'dhash', 'ahash', 'whash']:
                return self._calculate_hash_similarity(image1_path, image2_path, method)
            elif method == 'histogram':
                return self._calculate_histogram_similarity(image1_path, image2_path)
            else:
                logger.error(f"Unsupported similarity method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return None
    
    def _calculate_hash_similarity(self, image1_path: Path, image2_path: Path, 
                                 method: str) -> Optional[float]:
        """Calculate similarity using perceptual hashing."""
        hash1 = self.calculate_perceptual_hash(image1_path, method)
        hash2 = self.calculate_perceptual_hash(image2_path, method)
        
        if hash1 is None or hash2 is None:
            return None
        
        # Convert hash strings to imagehash objects for comparison
        if method == 'phash':
            hash1_obj = imagehash.hex_to_hash(hash1)
            hash2_obj = imagehash.hex_to_hash(hash2)
        elif method == 'dhash':
            hash1_obj = imagehash.hex_to_hash(hash1)
            hash2_obj = imagehash.hex_to_hash(hash2)
        elif method == 'ahash':
            hash1_obj = imagehash.hex_to_hash(hash1)
            hash2_obj = imagehash.hex_to_hash(hash2)
        elif method == 'whash':
            hash1_obj = imagehash.hex_to_hash(hash1)
            hash2_obj = imagehash.hex_to_hash(hash2)
        
        # Calculate Hamming distance
        hamming_distance = hash1_obj - hash2_obj
        
        # Convert to similarity score (0.0 to 1.0)
        # Lower Hamming distance = higher similarity
        max_distance = len(hash1_obj.hash.flatten())
        similarity = 1.0 - (hamming_distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_histogram_similarity(self, image1_path: Path, image2_path: Path) -> Optional[float]:
        """Calculate similarity using color histograms."""
        hist1 = self.calculate_histogram_features(image1_path)
        hist2 = self.calculate_histogram_features(image2_path)
        
        if hist1 is None or hist2 is None:
            return None
        
        # Calculate correlation coefficient
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Correlation ranges from -1 to 1, convert to 0 to 1
        similarity = (correlation + 1.0) / 2.0
        
        return max(0.0, min(1.0, similarity))
    
    def find_similar_images(self, target_image: Path, candidate_images: List[Path], 
                          method: str = 'phash', threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find similar images from a list of candidates.
        
        Args:
            target_image: Path to target image
            candidate_images: List of candidate image paths
            method: Similarity method to use
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of similar images with similarity scores
        """
        threshold = threshold or self.similarity_threshold
        similar_images = []
        
        logger.info(f"Finding similar images for {target_image.name} using {method}")
        
        for candidate in candidate_images:
            if candidate == target_image:
                continue  # Skip self-comparison
            
            similarity = self.calculate_similarity_score(target_image, candidate, method)
            
            if similarity is not None and similarity >= threshold:
                similar_images.append({
                    'image_path': str(candidate),
                    'similarity_score': similarity,
                    'method': method,
                    'target_image': str(target_image)
                })
        
        # Sort by similarity score (descending)
        similar_images.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Found {len(similar_images)} similar images above threshold {threshold}")
        return similar_images
    
    def batch_similarity_analysis(self, images: List[Path], method: str = 'phash') -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform similarity analysis on a batch of images.
        
        Args:
            images: List of image paths
            method: Similarity method to use
            
        Returns:
            Dictionary mapping each image to its similar images
        """
        results = {}
        total_comparisons = len(images) * (len(images) - 1) // 2
        
        logger.info(f"Starting batch similarity analysis for {len(images)} images ({total_comparisons} comparisons)")
        
        for i, target_image in enumerate(images):
            logger.debug(f"Processing image {i+1}/{len(images)}: {target_image.name}")
            
            # Find similar images among remaining candidates
            candidates = images[i+1:]  # Avoid duplicate comparisons
            similar = self.find_similar_images(target_image, candidates, method)
            
            results[str(target_image)] = similar
        
        logger.info("Batch similarity analysis completed")
        return results
    
    def get_image_features(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive features from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with various image features
        """
        features = {
            'image_path': str(image_path),
            'hashes': {},
            'histogram_features': None,
            'basic_info': {}
        }
        
        try:
            # Calculate all hash types
            for method in ['phash', 'dhash', 'ahash', 'whash']:
                features['hashes'][method] = self.calculate_perceptual_hash(image_path, method)
            
            # Calculate histogram features
            features['histogram_features'] = self.calculate_histogram_features(image_path)
            
            # Get basic image info
            if image_path.exists():
                with Image.open(image_path) as img:
                    features['basic_info'] = {
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format,
                        'file_size': image_path.stat().st_size
                    }
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
        
        return features
    
    def compare_with_multiple_methods(self, image1_path: Path, image2_path: Path) -> Dict[str, float]:
        """
        Compare two images using multiple similarity methods.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dictionary with similarity scores for each method
        """
        results = {}
        
        for method in self.supported_methods:
            score = self.calculate_similarity_score(image1_path, image2_path, method)
            if score is not None:
                results[method] = score
        
        return results
