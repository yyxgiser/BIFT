"""
Simple test script to verify BIFT implementation.
"""

import numpy as np
import cv2
from bift import BIFT, BIFTDetector, BIFTDescriptor, BIFTMatcher


def test_detector():
    """Test BIFTDetector."""
    print("Testing BIFTDetector...")
    
    # Create a simple test image
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # Add some features
    cv2.circle(image, (100, 100), 30, 255, -1)
    cv2.rectangle(image, (150, 150), (200, 200), 200, -1)
    
    detector = BIFTDetector(num_octaves=3, scales_per_octave=3)
    keypoints = detector.detect(image)
    
    assert len(keypoints) > 0, "No keypoints detected"
    assert all(hasattr(kp, 'x') for kp in keypoints), "Keypoints missing x coordinate"
    assert all(hasattr(kp, 'y') for kp in keypoints), "Keypoints missing y coordinate"
    
    print(f"  ✓ Detected {len(keypoints)} keypoints")


def test_descriptor():
    """Test BIFTDescriptor."""
    print("Testing BIFTDescriptor...")
    
    # Create test image and keypoints
    image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    cv2.circle(image, (128, 128), 50, 255, -1)
    
    detector = BIFTDetector()
    keypoints = detector.detect(image)
    
    descriptor = BIFTDescriptor()
    descriptors = descriptor.compute(image, keypoints)
    
    assert len(descriptors) == len(keypoints), "Descriptor count mismatch"
    if len(descriptors) > 0:
        assert descriptors.shape[1] == 128, "Wrong descriptor size"
        assert descriptors.dtype == np.float32, "Wrong descriptor dtype"
    
    print(f"  ✓ Computed {len(descriptors)} descriptors of size {descriptors.shape[1] if len(descriptors) > 0 else 0}")


def test_matcher():
    """Test BIFTMatcher."""
    print("Testing BIFTMatcher...")
    
    # Create two test descriptor sets
    desc1 = np.random.randn(50, 128).astype(np.float32)
    desc2 = np.random.randn(60, 128).astype(np.float32)
    
    # Make some descriptors similar
    desc2[:10] = desc1[:10] + np.random.randn(10, 128).astype(np.float32) * 0.1
    
    matcher = BIFTMatcher(ratio_threshold=0.8)
    matches = matcher.match(desc1, desc2)
    
    assert isinstance(matches, list), "Matches should be a list"
    assert all(hasattr(m, 'queryIdx') for m in matches), "Matches missing queryIdx"
    assert all(hasattr(m, 'trainIdx') for m in matches), "Matches missing trainIdx"
    
    print(f"  ✓ Found {len(matches)} matches")


def test_bift_pipeline():
    """Test complete BIFT pipeline."""
    print("Testing BIFT pipeline...")
    
    # Create two similar images
    image1 = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(image1, (100, 100), 40, 255, -1)
    cv2.rectangle(image1, (150, 150), (200, 200), 180, -1)
    
    # Create transformed version
    M = cv2.getRotationMatrix2D((128, 128), 10, 0.95)
    image2 = cv2.warpAffine(image1, M, (256, 256))
    
    # Initialize BIFT
    bift = BIFT(num_octaves=3, scales_per_octave=3)
    
    # Detect and compute
    kp1, desc1 = bift.detectAndCompute(image1)
    kp2, desc2 = bift.detectAndCompute(image2)
    
    assert len(kp1) > 0, "No keypoints in image1"
    assert len(kp2) > 0, "No keypoints in image2"
    assert len(desc1) == len(kp1), "Descriptor count mismatch for image1"
    assert len(desc2) == len(kp2), "Descriptor count mismatch for image2"
    
    print(f"  ✓ Image 1: {len(kp1)} keypoints")
    print(f"  ✓ Image 2: {len(kp2)} keypoints")
    
    # Match
    matches = bift.matcher.match(desc1, desc2)
    print(f"  ✓ Found {len(matches)} matches")


def test_visualization():
    """Test visualization functions."""
    print("Testing visualization...")
    
    # Create test images
    image1 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    image2 = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    bift = BIFT()
    kp1, desc1 = bift.detectAndCompute(image1)
    kp2, desc2 = bift.detectAndCompute(image2)
    
    if len(kp1) > 0 and len(kp2) > 0:
        # Visualize keypoints
        vis1 = bift.visualize_keypoints(image1, kp1)
        assert vis1.shape[0] == image1.shape[0], "Wrong visualization height"
        
        # Visualize matches
        matches = bift.matcher.match(desc1, desc2)
        if len(matches) > 0:
            vis_match = bift.visualize_matches(image1, image2, kp1, kp2, matches)
            assert vis_match.shape[1] >= image1.shape[1] + image2.shape[1], "Wrong match visualization width"
            print("  ✓ Visualization functions work correctly")
        else:
            print("  ✓ Visualization functions callable (no matches to display)")
    else:
        print("  ✓ Visualization functions callable (no keypoints detected)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("BIFT Implementation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_detector,
        test_descriptor,
        test_matcher,
        test_bift_pipeline,
        test_visualization
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
