"""
BIFT Matcher: Feature matching with ratio test
"""

import numpy as np
from typing import List, Tuple, Optional


class Match:
    """Represents a match between two descriptors."""
    
    def __init__(self, query_idx: int, train_idx: int, distance: float):
        self.queryIdx = query_idx
        self.trainIdx = train_idx
        self.distance = distance
    
    def __repr__(self):
        return f"Match(queryIdx={self.queryIdx}, trainIdx={self.trainIdx}, distance={self.distance:.4f})"


class BIFTMatcher:
    """
    BIFT Feature Matcher using ratio test for robust matching.
    
    Implements Lowe's ratio test to eliminate ambiguous matches.
    """
    
    def __init__(self, 
                 ratio_threshold: float = 0.8,
                 distance_metric: str = 'euclidean'):
        """
        Initialize BIFT matcher.
        
        Args:
            ratio_threshold: Ratio test threshold (default 0.8)
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        self.ratio_threshold = ratio_threshold
        self.distance_metric = distance_metric
    
    def match(self, 
              descriptors1: np.ndarray, 
              descriptors2: np.ndarray) -> List[Match]:
        """
        Match descriptors between two images.
        
        Args:
            descriptors1: Descriptors from first image (N x D)
            descriptors2: Descriptors from second image (M x D)
            
        Returns:
            List of matches
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Compute pairwise distances
        distances = self._compute_distances(descriptors1, descriptors2)
        
        # Find matches using ratio test
        matches = self._ratio_test(distances)
        
        return matches
    
    def knn_match(self,
                  descriptors1: np.ndarray,
                  descriptors2: np.ndarray,
                  k: int = 2) -> List[List[Match]]:
        """
        Find k nearest neighbors for each descriptor.
        
        Args:
            descriptors1: Descriptors from first image (N x D)
            descriptors2: Descriptors from second image (M x D)
            k: Number of nearest neighbors
            
        Returns:
            List of k matches for each descriptor
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Compute pairwise distances
        distances = self._compute_distances(descriptors1, descriptors2)
        
        # Find k nearest neighbors
        knn_matches = []
        for i in range(len(descriptors1)):
            # Sort distances and get k smallest
            sorted_indices = np.argsort(distances[i])
            k_nearest = sorted_indices[:k]
            
            matches_for_query = []
            for idx in k_nearest:
                match = Match(i, int(idx), distances[i, idx])
                matches_for_query.append(match)
            
            knn_matches.append(matches_for_query)
        
        return knn_matches
    
    def _compute_distances(self, 
                          descriptors1: np.ndarray, 
                          descriptors2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between descriptors.
        
        Args:
            descriptors1: First set of descriptors (N x D)
            descriptors2: Second set of descriptors (M x D)
            
        Returns:
            Distance matrix (N x M)
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            # Using broadcasting for efficient computation
            diff = descriptors1[:, np.newaxis, :] - descriptors2[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        elif self.distance_metric == 'cosine':
            # Cosine distance
            # Normalize descriptors
            norm1 = np.linalg.norm(descriptors1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(descriptors2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norm1 = np.where(norm1 == 0, 1, norm1)
            norm2 = np.where(norm2 == 0, 1, norm2)
            
            desc1_normalized = descriptors1 / norm1
            desc2_normalized = descriptors2 / norm2
            
            # Compute cosine similarity
            similarity = np.dot(desc1_normalized, desc2_normalized.T)
            
            # Convert to distance
            distances = 1 - similarity
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def _ratio_test(self, distances: np.ndarray) -> List[Match]:
        """
        Apply ratio test to filter matches.
        
        Args:
            distances: Distance matrix (N x M)
            
        Returns:
            List of good matches
        """
        matches = []
        
        for i in range(distances.shape[0]):
            # Sort distances for this query
            sorted_indices = np.argsort(distances[i])
            
            # Get two nearest neighbors
            if len(sorted_indices) < 2:
                continue
            
            nearest_idx = sorted_indices[0]
            second_nearest_idx = sorted_indices[1]
            
            nearest_dist = distances[i, nearest_idx]
            second_nearest_dist = distances[i, second_nearest_idx]
            
            # Apply ratio test
            if second_nearest_dist > 1e-10:  # Avoid division by zero
                ratio = nearest_dist / second_nearest_dist
                
                if ratio < self.ratio_threshold:
                    match = Match(i, int(nearest_idx), nearest_dist)
                    matches.append(match)
        
        return matches
    
    def filter_matches_by_homography(self,
                                     matches: List[Match],
                                     keypoints1: List,
                                     keypoints2: List,
                                     ransac_threshold: float = 3.0) -> Tuple[List[Match], Optional[np.ndarray]]:
        """
        Filter matches using RANSAC-based homography estimation.
        
        Args:
            matches: List of matches
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            ransac_threshold: RANSAC reprojection threshold
            
        Returns:
            Tuple of (filtered matches, homography matrix)
        """
        if len(matches) < 4:
            return matches, None
        
        # Extract matched point coordinates
        points1 = np.float32([keypoints1[m.queryIdx].pt() for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt() for m in matches])
        
        # Estimate homography using RANSAC
        try:
            import cv2
            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_threshold)
            
            if H is None:
                return matches, None
            
            # Filter matches based on inliers
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
            
            return inlier_matches, H
        except Exception:
            return matches, None
