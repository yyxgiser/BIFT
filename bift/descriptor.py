"""
BIFT Descriptor: Biological-inspired feature descriptor computation
"""

import numpy as np
import cv2
from typing import List, Tuple
from .detector import Keypoint
from .utils import normalize_image, compute_gradient


class BIFTDescriptor:
    """
    BIFT Descriptor generator using gradient-based histograms.
    
    Creates rotation-invariant and illumination-invariant descriptors
    inspired by biological visual cortex processing.
    """
    
    def __init__(self,
                 descriptor_size: int = 128,
                 num_bins: int = 8,
                 patch_size: int = 16,
                 num_spatial_bins: int = 4):
        """
        Initialize BIFT descriptor.
        
        Args:
            descriptor_size: Size of the descriptor vector (default 128)
            num_bins: Number of orientation bins
            patch_size: Size of the patch around keypoint
            num_spatial_bins: Number of spatial bins (4x4 grid)
        """
        self.descriptor_size = descriptor_size
        self.num_bins = num_bins
        self.patch_size = patch_size
        self.num_spatial_bins = num_spatial_bins
    
    def compute(self, image: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
        """
        Compute descriptors for keypoints.
        
        Args:
            image: Input image
            keypoints: List of keypoints
            
        Returns:
            Array of descriptors (N x descriptor_size)
        """
        # Normalize image
        gray = normalize_image(image)
        
        # Compute gradients
        magnitude, orientation = compute_gradient(gray)
        
        descriptors = []
        
        for kp in keypoints:
            # Assign dominant orientation if not already set
            if kp.angle < 0:
                kp.angle = self._compute_orientation(
                    magnitude, orientation, int(kp.y), int(kp.x), kp.size
                )
            
            # Compute descriptor
            descriptor = self._compute_descriptor(
                magnitude, orientation, kp
            )
            
            if descriptor is not None:
                descriptors.append(descriptor)
            else:
                # Add zero descriptor if computation failed
                descriptors.append(np.zeros(self.descriptor_size))
        
        if len(descriptors) == 0:
            return np.array([])
        
        return np.array(descriptors, dtype=np.float32)
    
    def _compute_orientation(self,
                            magnitude: np.ndarray,
                            orientation: np.ndarray,
                            y: int,
                            x: int,
                            size: float) -> float:
        """
        Compute dominant orientation for rotation invariance.
        
        Args:
            magnitude: Gradient magnitude
            orientation: Gradient orientation
            y: Keypoint y coordinate
            x: Keypoint x coordinate
            size: Keypoint size (scale)
            
        Returns:
            Dominant orientation in degrees
        """
        # Define region around keypoint
        radius = int(1.5 * size / 2)
        
        # Create histogram of orientations
        hist = np.zeros(36)  # 36 bins for 10-degree intervals
        
        height, width = magnitude.shape
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py = y + dy
                px = x + dx
                
                # Check bounds
                if py < 0 or py >= height or px < 0 or px >= width:
                    continue
                
                # Compute distance from keypoint
                dist = np.sqrt(dy * dy + dx * dx)
                if dist > radius:
                    continue
                
                # Get gradient magnitude and orientation
                mag = magnitude[py, px]
                ori = orientation[py, px]
                
                # Weight by Gaussian and magnitude
                sigma = size / 2
                weight = np.exp(-(dist * dist) / (2 * sigma * sigma)) * mag
                
                # Convert orientation to histogram bin
                bin_idx = int(np.round((ori + np.pi) / (2 * np.pi) * 36)) % 36
                hist[bin_idx] += weight
        
        # Smooth histogram
        hist_smooth = np.zeros_like(hist)
        for i in range(36):
            prev_idx = (i - 1) % 36
            next_idx = (i + 1) % 36
            hist_smooth[i] = (hist[prev_idx] + hist[i] + hist[next_idx]) / 3.0
        
        # Find dominant orientation (peak)
        peak_idx = np.argmax(hist_smooth)
        
        # Convert to degrees
        angle = (peak_idx * 10.0) % 360
        
        return angle
    
    def _compute_descriptor(self,
                           magnitude: np.ndarray,
                           orientation: np.ndarray,
                           keypoint: Keypoint) -> np.ndarray:
        """
        Compute descriptor for a keypoint.
        
        Args:
            magnitude: Gradient magnitude
            orientation: Gradient orientation
            keypoint: Keypoint
            
        Returns:
            Descriptor vector
        """
        x, y = int(keypoint.x), int(keypoint.y)
        angle = keypoint.angle
        size = keypoint.size
        
        # Create rotation matrix for orientation normalization
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        
        # Initialize descriptor
        hist_width = self.num_spatial_bins
        descriptor_bins = hist_width * hist_width * self.num_bins
        descriptor = np.zeros(descriptor_bins)
        
        # Half patch size
        half_size = self.patch_size // 2
        
        # Bin width for spatial histograms
        bin_width = self.patch_size / hist_width
        
        height, width = magnitude.shape
        
        # Iterate over patch
        for dy in range(-half_size, half_size):
            for dx in range(-half_size, half_size):
                # Rotate coordinates
                rot_x = cos_angle * dx - sin_angle * dy
                rot_y = sin_angle * dx + cos_angle * dy
                
                # Get pixel position
                px = int(x + dx)
                py = int(y + dy)
                
                # Check bounds
                if px < 0 or px >= width or py < 0 or py >= height:
                    continue
                
                # Get gradient
                mag = magnitude[py, px]
                ori = orientation[py, px]
                
                # Rotate orientation
                ori_rot = ori - angle_rad
                
                # Gaussian weighting
                sigma = self.patch_size / 2
                dist = np.sqrt(dx * dx + dy * dy)
                weight = np.exp(-(dist * dist) / (2 * sigma * sigma)) * mag
                
                # Find spatial bin
                bin_x = (rot_x + half_size) / bin_width
                bin_y = (rot_y + half_size) / bin_width
                
                # Skip if outside patch
                if bin_x < 0 or bin_x >= hist_width or bin_y < 0 or bin_y >= hist_width:
                    continue
                
                # Trilinear interpolation for spatial bins
                x0 = int(np.floor(bin_x))
                y0 = int(np.floor(bin_y))
                
                if x0 < 0 or x0 >= hist_width - 1 or y0 < 0 or y0 >= hist_width - 1:
                    continue
                
                # Orientation bin
                ori_bin = (ori_rot + np.pi) / (2 * np.pi) * self.num_bins
                ori_bin0 = int(np.floor(ori_bin)) % self.num_bins
                ori_bin1 = (ori_bin0 + 1) % self.num_bins
                
                # Interpolation weights
                dx_interp = bin_x - x0
                dy_interp = bin_y - y0
                dori = ori_bin - np.floor(ori_bin)
                
                # Distribute to adjacent bins
                for i in range(2):
                    for j in range(2):
                        spatial_weight = (1 - abs(i - dx_interp)) * (1 - abs(j - dy_interp))
                        
                        # Bin indices
                        bx = x0 + i
                        by = y0 + j
                        
                        if bx >= 0 and bx < hist_width and by >= 0 and by < hist_width:
                            # Add to both orientation bins
                            idx0 = (by * hist_width + bx) * self.num_bins + ori_bin0
                            idx1 = (by * hist_width + bx) * self.num_bins + ori_bin1
                            
                            descriptor[idx0] += weight * spatial_weight * (1 - dori)
                            descriptor[idx1] += weight * spatial_weight * dori
        
        # Normalize descriptor for illumination invariance
        descriptor = self._normalize_descriptor(descriptor)
        
        return descriptor
    
    def _normalize_descriptor(self, descriptor: np.ndarray) -> np.ndarray:
        """
        Normalize descriptor for illumination invariance.
        
        Args:
            descriptor: Raw descriptor
            
        Returns:
            Normalized descriptor
        """
        # L2 normalization
        norm = np.linalg.norm(descriptor)
        if norm > 1e-10:
            descriptor = descriptor / norm
        
        # Threshold to reduce influence of large gradients
        threshold = 0.2
        descriptor = np.clip(descriptor, 0, threshold)
        
        # Renormalize
        norm = np.linalg.norm(descriptor)
        if norm > 1e-10:
            descriptor = descriptor / norm
        
        # Scale to [0, 255] for integer representation if needed
        # descriptor = (descriptor * 512).astype(np.float32)
        
        return descriptor
