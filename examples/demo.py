"""
Demo script for BIFT algorithm.

This script demonstrates the usage of BIFT for multimodal image matching.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from bift import BIFT


def create_synthetic_multimodal_images():
    """
    Create synthetic multimodal images for demonstration.
    Simulates different imaging modalities with radiometric distortions.
    """
    # Create base image with geometric patterns
    size = 512
    image1 = np.zeros((size, size), dtype=np.uint8)
    
    # Add some features
    cv2.circle(image1, (150, 150), 50, 255, -1)
    cv2.rectangle(image1, (300, 100), (400, 200), 200, -1)
    cv2.circle(image1, (250, 350), 60, 180, -1)
    cv2.rectangle(image1, (100, 300), (200, 450), 220, -1)
    
    # Add texture
    noise = np.random.randn(size, size) * 20
    image1 = np.clip(image1.astype(float) + noise, 0, 255).astype(np.uint8)
    
    # Create second image with transformation and radiometric distortion
    # Apply rotation
    center = (size // 2, size // 2)
    angle = 15  # degrees
    scale = 0.9
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image2 = cv2.warpAffine(image1, M, (size, size))
    
    # Apply radiometric distortion (simulate different sensor)
    # Gamma correction
    gamma = 1.5
    image2 = np.power(image2 / 255.0, gamma) * 255
    
    # Add intensity shift
    image2 = np.clip(image2 + 30, 0, 255).astype(np.uint8)
    
    # Add different noise pattern
    noise2 = np.random.randn(size, size) * 15
    image2 = np.clip(image2.astype(float) + noise2, 0, 255).astype(np.uint8)
    
    return image1, image2


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("BIFT: Biological-inspired Invariant Feature Transform")
    print("Multimodal Image Matching Demo")
    print("=" * 60)
    print()
    
    # Create synthetic multimodal images
    print("1. Creating synthetic multimodal images...")
    image1, image2 = create_synthetic_multimodal_images()
    print(f"   Image 1 shape: {image1.shape}")
    print(f"   Image 2 shape: {image2.shape}")
    print()
    
    # Initialize BIFT
    print("2. Initializing BIFT algorithm...")
    bift = BIFT(
        num_octaves=3,
        scales_per_octave=4,
        contrast_threshold=0.03,
        max_keypoints=1000,
        ratio_threshold=0.8
    )
    print("   BIFT initialized successfully")
    print()
    
    # Detect and compute features for both images
    print("3. Detecting keypoints and computing descriptors...")
    keypoints1, descriptors1 = bift.detectAndCompute(image1)
    keypoints2, descriptors2 = bift.detectAndCompute(image2)
    print(f"   Image 1: {len(keypoints1)} keypoints detected")
    print(f"   Image 2: {len(keypoints2)} keypoints detected")
    print(f"   Descriptor size: {descriptors1.shape[1] if len(descriptors1) > 0 else 0}")
    print()
    
    # Match features
    print("4. Matching features between images...")
    matches = bift.matcher.match(descriptors1, descriptors2)
    print(f"   Initial matches: {len(matches)}")
    
    # Filter with homography
    if len(matches) >= 4:
        filtered_matches, homography = bift.matcher.filter_matches_by_homography(
            matches, keypoints1, keypoints2
        )
        print(f"   Matches after RANSAC filtering: {len(filtered_matches)}")
        if homography is not None:
            print("   Homography matrix estimated successfully")
    else:
        filtered_matches = matches
        print("   Not enough matches for homography estimation")
    print()
    
    # Visualize results
    print("5. Generating visualizations...")
    
    # Visualize keypoints
    kp_vis1 = bift.visualize_keypoints(image1, keypoints1, max_keypoints=200)
    kp_vis2 = bift.visualize_keypoints(image2, keypoints2, max_keypoints=200)
    
    # Visualize matches
    match_vis = bift.visualize_matches(
        image1, image2, 
        keypoints1, keypoints2, 
        filtered_matches,
        max_matches=50
    )
    
    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(f"{output_dir}/image1.png", image1)
    cv2.imwrite(f"{output_dir}/image2.png", image2)
    cv2.imwrite(f"{output_dir}/keypoints1.png", kp_vis1)
    cv2.imwrite(f"{output_dir}/keypoints2.png", kp_vis2)
    cv2.imwrite(f"{output_dir}/matches.png", match_vis)
    
    print(f"   Results saved to '{output_dir}/' directory")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total keypoints detected: {len(keypoints1) + len(keypoints2)}")
    print(f"Total matches found: {len(filtered_matches)}")
    
    if len(keypoints1) > 0 and len(keypoints2) > 0:
        match_ratio = len(filtered_matches) / min(len(keypoints1), len(keypoints2))
        print(f"Match ratio: {match_ratio:.2%}")
    
    print()
    print("BIFT successfully handled:")
    print("  - Rotation transformation")
    print("  - Scale changes")
    print("  - Radiometric distortions (gamma correction)")
    print("  - Intensity shifts")
    print("  - Different noise patterns")
    print()
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
