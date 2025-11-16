"""
Parameter tuning example for BIFT algorithm.

This script demonstrates how different parameters affect BIFT performance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from bift import BIFT


def create_test_image_pair():
    """Create a pair of test images with known transformation."""
    size = 400
    image1 = np.zeros((size, size), dtype=np.uint8)
    
    # Add features
    cv2.circle(image1, (100, 100), 50, 255, -1)
    cv2.rectangle(image1, (250, 150), (350, 250), 200, -1)
    cv2.circle(image1, (200, 300), 40, 180, -1)
    cv2.circle(image1, (300, 300), 35, 220, -1)
    
    # Transform: rotation and scale
    center = (size // 2, size // 2)
    M = cv2.getRotationMatrix2D(center, 25, 0.9)
    image2 = cv2.warpAffine(image1, M, (size, size))
    
    # Add radiometric distortion
    image2 = np.power(image2 / 255.0, 1.3) * 255
    image2 = image2.astype(np.uint8)
    
    return image1, image2


def test_parameter_configuration(image1, image2, config_name, **params):
    """Test a parameter configuration."""
    print(f"\nTesting: {config_name}")
    print(f"  Parameters: {params}")
    
    bift = BIFT(**params)
    kp1, kp2, matches, H = bift.match(image1, image2)
    
    print(f"  Results:")
    print(f"    - Keypoints: {len(kp1)} and {len(kp2)}")
    print(f"    - Matches: {len(matches)}")
    print(f"    - Homography: {'Yes' if H is not None else 'No'}")
    
    return len(kp1), len(kp2), len(matches)


def main():
    """Main parameter tuning demonstration."""
    print("=" * 70)
    print("BIFT Parameter Tuning Demonstration")
    print("=" * 70)
    
    # Create test images
    print("\nCreating test image pair...")
    image1, image2 = create_test_image_pair()
    
    # Test different configurations
    configurations = [
        ("Default", {}),
        
        ("High sensitivity (more keypoints)", {
            "contrast_threshold": 0.02,
            "max_keypoints": 5000
        }),
        
        ("Low sensitivity (fewer keypoints)", {
            "contrast_threshold": 0.06,
            "max_keypoints": 500
        }),
        
        ("More scales (better scale invariance)", {
            "num_octaves": 5,
            "scales_per_octave": 6
        }),
        
        ("Fewer scales (faster)", {
            "num_octaves": 3,
            "scales_per_octave": 3
        }),
        
        ("Strict matching", {
            "ratio_threshold": 0.6
        }),
        
        ("Lenient matching", {
            "ratio_threshold": 0.9
        }),
        
        ("Balanced (recommended)", {
            "num_octaves": 4,
            "scales_per_octave": 5,
            "contrast_threshold": 0.04,
            "max_keypoints": 2000,
            "ratio_threshold": 0.75
        })
    ]
    
    results = []
    for config_name, params in configurations:
        kp1, kp2, matches = test_parameter_configuration(
            image1, image2, config_name, **params
        )
        results.append((config_name, kp1, kp2, matches))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'KP1':>6} {'KP2':>6} {'Matches':>8}")
    print("-" * 70)
    for config_name, kp1, kp2, matches in results:
        print(f"{config_name:<35} {kp1:>6} {kp2:>6} {matches:>8}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. For fast processing: Use fewer octaves and scales
   - num_octaves=3, scales_per_octave=3

2. For maximum accuracy: Use more octaves and scales
   - num_octaves=5, scales_per_octave=6

3. For high-texture images: Lower contrast threshold
   - contrast_threshold=0.02

4. For low-texture images: Higher contrast threshold
   - contrast_threshold=0.06

5. For strict matching: Lower ratio threshold
   - ratio_threshold=0.6-0.7

6. For lenient matching: Higher ratio threshold
   - ratio_threshold=0.8-0.9

7. Balanced default works well for most cases
    """)


if __name__ == "__main__":
    main()
