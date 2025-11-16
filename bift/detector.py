"""
BIFT Detector: Biological-inspired keypoint detection
"""

import numpy as np
import cv2
from typing import List, Tuple
from .utils import (normalize_image, build_scale_space, compute_dog_pyramid,
                    is_valid_keypoint, subpixel_refinement)


class Keypoint:
    """Represents a detected keypoint with its properties."""
    
    def __init__(self, x: float, y: float, size: float, angle: float = -1,
                 response: float = 0, octave: int = 0, class_id: int = -1):
        self.x = x
        self.y = y
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id
    
    def pt(self) -> Tuple[float, float]:
        """Return keypoint location as tuple."""
        return (self.x, self.y)
    
    def to_cv_keypoint(self) -> cv2.KeyPoint:
        """Convert to OpenCV KeyPoint format."""
        return cv2.KeyPoint(x=float(self.x), y=float(self.y), 
                           size=float(self.size), angle=float(self.angle),
                           response=float(self.response), octave=int(self.octave),
                           class_id=int(self.class_id))


class BIFTDetector:
    """
    BIFT Keypoint Detector using biological-inspired scale-space extrema detection.
    
    This detector uses Difference of Gaussians (DoG) to detect scale-invariant
    keypoints, inspired by biological vision systems that use center-surround
    receptive fields.
    """
    
    def __init__(self, 
                 num_octaves: int = 4,
                 scales_per_octave: int = 5,
                 sigma: float = 1.6,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0,
                 max_keypoints: int = 10000):
        """
        Initialize BIFT detector.
        
        Args:
            num_octaves: Number of octaves in scale space
            scales_per_octave: Number of scales per octave
            sigma: Initial Gaussian sigma
            contrast_threshold: Threshold for low contrast keypoint removal
            edge_threshold: Threshold for edge response removal
            max_keypoints: Maximum number of keypoints to detect
        """
        self.num_octaves = num_octaves
        self.scales_per_octave = scales_per_octave
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.max_keypoints = max_keypoints
    
    def detect(self, image: np.ndarray) -> List[Keypoint]:
        """
        Detect keypoints in the image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            List of detected keypoints
        """
        # Normalize image to handle radiometric distortions
        gray = normalize_image(image)
        
        # Build scale space
        scale_space = build_scale_space(
            gray, 
            self.num_octaves, 
            self.scales_per_octave, 
            self.sigma
        )
        
        # Compute DoG pyramid (biological-inspired edge detection)
        dog_pyramid = compute_dog_pyramid(scale_space)
        
        # Detect keypoints
        keypoints = self._find_scale_space_extrema(dog_pyramid, scale_space)
        
        # Limit number of keypoints
        if len(keypoints) > self.max_keypoints:
            keypoints.sort(key=lambda kp: kp.response, reverse=True)
            keypoints = keypoints[:self.max_keypoints]
        
        return keypoints
    
    def _find_scale_space_extrema(self, 
                                   dog_pyramid: List[List[np.ndarray]],
                                   scale_space: List[List[np.ndarray]]) -> List[Keypoint]:
        """
        Find scale-space extrema in DoG pyramid.
        
        Args:
            dog_pyramid: DoG pyramid
            scale_space: Scale space pyramid
            
        Returns:
            List of keypoints
        """
        keypoints = []
        
        for octave_idx, octave_dogs in enumerate(dog_pyramid):
            for scale_idx in range(1, len(octave_dogs) - 1):
                dog_prev = octave_dogs[scale_idx - 1]
                dog_curr = octave_dogs[scale_idx]
                dog_next = octave_dogs[scale_idx + 1]
                
                # Find extrema
                extrema = self._find_extrema_in_scale(dog_prev, dog_curr, dog_next)
                
                for y, x in extrema:
                    # Refine keypoint location
                    refined_x, refined_y, refined_scale = subpixel_refinement(
                        dog_pyramid, octave_idx, scale_idx, y, x
                    )
                    
                    # Check if refinement moved the point too much
                    if abs(refined_x - x) > 1.5 or abs(refined_y - y) > 1.5:
                        continue
                    
                    # Convert to original image coordinates
                    scale_factor = 2 ** octave_idx
                    orig_x = refined_x * scale_factor
                    orig_y = refined_y * scale_factor
                    
                    # Check if valid
                    if not is_valid_keypoint(int(orig_x), int(orig_y), 
                                            scale_space[0][0].shape, border=10):
                        continue
                    
                    # Compute response (contrast)
                    response = abs(dog_curr[y, x])
                    
                    # Filter low contrast keypoints
                    if response < self.contrast_threshold:
                        continue
                    
                    # Filter edge responses using Hessian
                    if not self._is_not_edge(dog_curr, y, x):
                        continue
                    
                    # Compute keypoint size (scale)
                    sigma_scale = self.sigma * (2 ** (scale_idx / self.scales_per_octave))
                    size = sigma_scale * scale_factor * 2
                    
                    keypoint = Keypoint(
                        x=orig_x,
                        y=orig_y,
                        size=size,
                        response=response,
                        octave=octave_idx
                    )
                    keypoints.append(keypoint)
        
        return keypoints
    
    def _find_extrema_in_scale(self, 
                               dog_prev: np.ndarray,
                               dog_curr: np.ndarray,
                               dog_next: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find local extrema in the current scale.
        
        Args:
            dog_prev: DoG image at previous scale
            dog_curr: DoG image at current scale
            dog_next: DoG image at next scale
            
        Returns:
            List of (y, x) coordinates of extrema
        """
        extrema = []
        height, width = dog_curr.shape
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                val = dog_curr[y, x]
                
                # Check if it's a maximum or minimum
                neighborhood = np.array([
                    dog_prev[y-1:y+2, x-1:x+2],
                    dog_curr[y-1:y+2, x-1:x+2],
                    dog_next[y-1:y+2, x-1:x+2]
                ])
                
                # Remove center point for comparison
                center_val = val
                neighborhood_flat = neighborhood.flatten()
                center_idx = len(neighborhood_flat) // 2
                neighborhood_flat = np.delete(neighborhood_flat, center_idx)
                
                # Check for extrema
                is_maximum = center_val > np.max(neighborhood_flat)
                is_minimum = center_val < np.min(neighborhood_flat)
                
                if is_maximum or is_minimum:
                    extrema.append((y, x))
        
        return extrema
    
    def _is_not_edge(self, dog: np.ndarray, y: int, x: int) -> bool:
        """
        Check if keypoint is not on an edge using principal curvature.
        
        Args:
            dog: DoG image
            y: Y coordinate
            x: X coordinate
            
        Returns:
            True if keypoint is not on an edge
        """
        # Compute Hessian matrix
        dxx = dog[y, x+1] + dog[y, x-1] - 2 * dog[y, x]
        dyy = dog[y+1, x] + dog[y-1, x] - 2 * dog[y, x]
        dxy = ((dog[y+1, x+1] - dog[y+1, x-1]) - 
               (dog[y-1, x+1] - dog[y-1, x-1])) / 4.0
        
        # Compute trace and determinant
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        # Avoid division by zero
        if abs(det) < 1e-10:
            return False
        
        # Check edge threshold
        # ratio of principal curvatures
        ratio = trace * trace / det
        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        
        return ratio < threshold
