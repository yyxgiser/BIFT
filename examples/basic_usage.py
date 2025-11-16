"""
Basic usage example for BIFT algorithm.

This script shows the simplest way to use BIFT for feature detection and matching.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from bift import BIFT


def main():
    """Basic usage demonstration."""
    
    # Create two test images
    print("Creating test images...")
    size = 400
    image1 = np.zeros((size, size), dtype=np.uint8)
    
    # Add features to first image
    cv2.circle(image1, (100, 100), 50, 255, -1)
    cv2.rectangle(image1, (250, 150), (350, 250), 200, -1)
    cv2.circle(image1, (200, 300), 40, 180, -1)
    
    # Create second image with rotation
    center = (size // 2, size // 2)
    M = cv2.getRotationMatrix2D(center, 20, 1.0)
    image2 = cv2.warpAffine(image1, M, (size, size))
    
    # Initialize BIFT
    print("Initializing BIFT...")
    bift = BIFT()
    
    # Method 1: Detect and compute separately
    print("\nMethod 1: Detect and compute separately")
    keypoints1, descriptors1 = bift.detectAndCompute(image1)
    keypoints2, descriptors2 = bift.detectAndCompute(image2)
    print(f"  Image 1: {len(keypoints1)} keypoints")
    print(f"  Image 2: {len(keypoints2)} keypoints")
    
    # Match features
    matches = bift.matcher.match(descriptors1, descriptors2)
    print(f"  Matches: {len(matches)}")
    
    # Method 2: All-in-one matching
    print("\nMethod 2: Complete matching pipeline")
    kp1, kp2, matches, homography = bift.match(image1, image2)
    print(f"  Keypoints: {len(kp1)} and {len(kp2)}")
    print(f"  Matches: {len(matches)}")
    if homography is not None:
        print(f"  Homography estimated: Yes")
    
    # Visualize
    print("\nGenerating visualization...")
    match_img = bift.visualize_matches(image1, image2, kp1, kp2, matches)
    
    # Save result
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/basic_matches.png', match_img)
    print("  Saved to 'output/basic_matches.png'")
    
    print("\nBasic usage demonstration completed!")


if __name__ == "__main__":
    main()
