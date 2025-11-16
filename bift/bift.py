"""
BIFT: Main interface for Biological-inspired Invariant Feature Transform
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from .detector import BIFTDetector, Keypoint
from .descriptor import BIFTDescriptor
from .matcher import BIFTMatcher, Match


class BIFT:
    """
    Complete BIFT pipeline for multimodal image matching.
    
    Combines biological-inspired detection, descriptor computation,
    and robust matching for handling radiometric distortions and
    spectral discrepancies in multimodal remote sensing images.
    """
    
    def __init__(self,
                 num_octaves: int = 4,
                 scales_per_octave: int = 5,
                 sigma: float = 1.6,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10.0,
                 max_keypoints: int = 10000,
                 descriptor_size: int = 128,
                 num_bins: int = 8,
                 patch_size: int = 16,
                 ratio_threshold: float = 0.8):
        """
        Initialize BIFT.
        
        Args:
            num_octaves: Number of octaves in scale space
            scales_per_octave: Number of scales per octave
            sigma: Initial Gaussian sigma
            contrast_threshold: Threshold for low contrast keypoint removal
            edge_threshold: Threshold for edge response removal
            max_keypoints: Maximum number of keypoints to detect
            descriptor_size: Size of descriptor vector
            num_bins: Number of orientation bins in descriptor
            patch_size: Size of patch around keypoint
            ratio_threshold: Ratio test threshold for matching
        """
        # Initialize detector
        self.detector = BIFTDetector(
            num_octaves=num_octaves,
            scales_per_octave=scales_per_octave,
            sigma=sigma,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            max_keypoints=max_keypoints
        )
        
        # Initialize descriptor
        self.descriptor = BIFTDescriptor(
            descriptor_size=descriptor_size,
            num_bins=num_bins,
            patch_size=patch_size
        )
        
        # Initialize matcher
        self.matcher = BIFTMatcher(
            ratio_threshold=ratio_threshold
        )
    
    def detectAndCompute(self, image: np.ndarray) -> Tuple[List[Keypoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Detect keypoints
        keypoints = self.detector.detect(image)
        
        # Compute descriptors
        descriptors = self.descriptor.compute(image, keypoints)
        
        return keypoints, descriptors
    
    def match(self,
              image1: np.ndarray,
              image2: np.ndarray,
              use_homography_filter: bool = True,
              ransac_threshold: float = 3.0) -> Tuple[List[Keypoint], List[Keypoint], List[Match], Optional[np.ndarray]]:
        """
        Match features between two images.
        
        Args:
            image1: First image
            image2: Second image
            use_homography_filter: Whether to filter matches using homography
            ransac_threshold: RANSAC threshold for homography filtering
            
        Returns:
            Tuple of (keypoints1, keypoints2, matches, homography)
        """
        # Detect and compute for both images
        keypoints1, descriptors1 = self.detectAndCompute(image1)
        keypoints2, descriptors2 = self.detectAndCompute(image2)
        
        # Match descriptors
        matches = self.matcher.match(descriptors1, descriptors2)
        
        # Filter matches using homography if requested
        homography = None
        if use_homography_filter and len(matches) >= 4:
            matches, homography = self.matcher.filter_matches_by_homography(
                matches, keypoints1, keypoints2, ransac_threshold
            )
        
        return keypoints1, keypoints2, matches, homography
    
    def visualize_matches(self,
                         image1: np.ndarray,
                         image2: np.ndarray,
                         keypoints1: List[Keypoint],
                         keypoints2: List[Keypoint],
                         matches: List[Match],
                         max_matches: int = 50) -> np.ndarray:
        """
        Visualize matches between two images.
        
        Args:
            image1: First image
            image2: Second image
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: List of matches
            max_matches: Maximum number of matches to display
            
        Returns:
            Visualization image
        """
        # Convert to color if grayscale
        if len(image1.shape) == 2:
            img1_vis = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        else:
            img1_vis = image1.copy()
        
        if len(image2.shape) == 2:
            img2_vis = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        else:
            img2_vis = image2.copy()
        
        # Ensure same height
        h1, w1 = img1_vis.shape[:2]
        h2, w2 = img2_vis.shape[:2]
        
        if h1 != h2:
            max_h = max(h1, h2)
            if h1 < max_h:
                pad = max_h - h1
                img1_vis = cv2.copyMakeBorder(img1_vis, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
            if h2 < max_h:
                pad = max_h - h2
                img2_vis = cv2.copyMakeBorder(img2_vis, 0, pad, 0, 0, cv2.BORDER_CONSTANT)
        
        # Concatenate images side by side
        vis_img = np.hstack([img1_vis, img2_vis])
        
        # Limit number of matches to display
        if len(matches) > max_matches:
            # Sort by distance and take best matches
            sorted_matches = sorted(matches, key=lambda m: m.distance)
            matches = sorted_matches[:max_matches]
        
        # Draw matches
        for match in matches:
            pt1 = keypoints1[match.queryIdx].pt()
            pt2 = keypoints2[match.trainIdx].pt()
            
            # Offset second point by width of first image
            pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
            pt1_int = (int(pt1[0]), int(pt1[1]))
            
            # Random color for each match
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw line
            cv2.line(vis_img, pt1_int, pt2_shifted, color, 1)
            
            # Draw circles
            cv2.circle(vis_img, pt1_int, 3, color, -1)
            cv2.circle(vis_img, pt2_shifted, 3, color, -1)
        
        return vis_img
    
    def visualize_keypoints(self,
                           image: np.ndarray,
                           keypoints: List[Keypoint],
                           max_keypoints: int = 500) -> np.ndarray:
        """
        Visualize keypoints on an image.
        
        Args:
            image: Input image
            keypoints: List of keypoints
            max_keypoints: Maximum number of keypoints to display
            
        Returns:
            Visualization image
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = image.copy()
        
        # Limit number of keypoints
        if len(keypoints) > max_keypoints:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_keypoints]
        
        # Draw keypoints
        for kp in keypoints:
            pt = (int(kp.x), int(kp.y))
            size = int(kp.size / 2)
            
            # Draw circle
            cv2.circle(vis_img, pt, size, (0, 255, 0), 1)
            
            # Draw orientation line if available
            if kp.angle >= 0:
                angle_rad = np.deg2rad(kp.angle)
                end_x = int(kp.x + size * np.cos(angle_rad))
                end_y = int(kp.y + size * np.sin(angle_rad))
                cv2.line(vis_img, pt, (end_x, end_y), (0, 0, 255), 1)
        
        return vis_img
