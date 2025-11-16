# BIFT: Biological-inspired Invariant Feature Transform

Robust Multimodal Image Matching via Biological-inspired Invariant Feature Transform

## Overview

BIFT is a novel feature detection and matching algorithm designed for multimodal remote sensing images (MRSI). It addresses the challenges of nonlinear radiometric distortions and spectral discrepancies that arise from diverse imaging mechanisms.

### Key Features

- **Biological-inspired Detection**: Uses scale-space extrema detection inspired by biological vision systems
- **Radiometric Invariance**: Robust to nonlinear radiometric distortions through normalization and descriptor design
- **Rotation Invariance**: Assigns dominant orientation to descriptors for rotation invariance
- **Scale Invariance**: Multi-scale detection using Difference of Gaussians (DoG) pyramid
- **Robust Matching**: Implements ratio test and RANSAC-based filtering for reliable matches

## Installation

### From source

```bash
git clone https://github.com/yyxgiser/BIFT.git
cd BIFT
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.6
- NumPy >= 1.19.0
- OpenCV >= 4.5.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0

### Verify Installation

```bash
# Check version
python -m bift --version

# Run tests
python test_bift.py

# Run demo
python examples/demo.py
```

## Quick Start

```python
import cv2
from bift import BIFT

# Load images
image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Initialize BIFT
bift = BIFT()

# Detect and match features
keypoints1, keypoints2, matches, homography = bift.match(image1, image2)

# Visualize results
match_img = bift.visualize_matches(image1, image2, keypoints1, keypoints2, matches)
cv2.imwrite('matches.png', match_img)

print(f"Found {len(matches)} matches")
```

## Usage Examples

### Basic Feature Detection

```python
from bift import BIFT

# Initialize BIFT detector
bift = BIFT(
    num_octaves=4,
    scales_per_octave=5,
    contrast_threshold=0.04,
    max_keypoints=5000
)

# Detect keypoints and compute descriptors
keypoints, descriptors = bift.detectAndCompute(image)
print(f"Detected {len(keypoints)} keypoints")
```

### Custom Matching

```python
from bift import BIFTDetector, BIFTDescriptor, BIFTMatcher

# Create components separately
detector = BIFTDetector(num_octaves=3, scales_per_octave=4)
descriptor = BIFTDescriptor(descriptor_size=128)
matcher = BIFTMatcher(ratio_threshold=0.8)

# Detect keypoints
kp1 = detector.detect(image1)
kp2 = detector.detect(image2)

# Compute descriptors
desc1 = descriptor.compute(image1, kp1)
desc2 = descriptor.compute(image2, kp2)

# Match
matches = matcher.match(desc1, desc2)
```

### Running the Demo

```bash
cd examples
python demo.py
```

This will create synthetic multimodal images with radiometric distortions and demonstrate the BIFT matching pipeline.

## Algorithm Description

### 1. Feature Detection

BIFT uses a biological-inspired approach to detect scale-invariant keypoints:

1. **Scale Space Construction**: Builds a pyramid of Gaussian-blurred images at different scales
2. **DoG Computation**: Computes Difference of Gaussians (DoG), mimicking center-surround receptive fields in biological vision
3. **Extrema Detection**: Finds local extrema in scale space
4. **Keypoint Refinement**: Uses quadratic interpolation for subpixel accuracy
5. **Filtering**: Removes low-contrast and edge responses

### 2. Descriptor Computation

The descriptor design ensures invariance to various transformations:

1. **Orientation Assignment**: Computes dominant orientation from gradient histogram for rotation invariance
2. **Gradient Computation**: Extracts gradient magnitude and orientation in normalized patch
3. **Histogram Creation**: Creates spatial histogram of gradients (4x4 grid with 8 orientation bins = 128D descriptor)
4. **Normalization**: L2 normalization with clipping for illumination invariance

### 3. Feature Matching

Robust matching using:

1. **Distance Computation**: Euclidean or cosine distance between descriptors
2. **Ratio Test**: Lowe's ratio test to eliminate ambiguous matches
3. **RANSAC Filtering**: Optional homography-based outlier removal

## Parameters

### BIFT Constructor

- `num_octaves` (int): Number of octaves in scale space (default: 4)
- `scales_per_octave` (int): Scales per octave (default: 5)
- `sigma` (float): Initial Gaussian sigma (default: 1.6)
- `contrast_threshold` (float): Contrast threshold for keypoint filtering (default: 0.04)
- `edge_threshold` (float): Edge response threshold (default: 10.0)
- `max_keypoints` (int): Maximum keypoints to detect (default: 10000)
- `descriptor_size` (int): Descriptor vector size (default: 128)
- `num_bins` (int): Orientation bins in descriptor (default: 8)
- `patch_size` (int): Patch size around keypoint (default: 16)
- `ratio_threshold` (float): Ratio test threshold (default: 0.8)

## Applications

BIFT is particularly suited for:

- **Multimodal Remote Sensing**: Matching images from different sensors (optical, SAR, thermal, etc.)
- **Change Detection**: Detecting changes in multi-temporal images with different acquisition conditions
- **Image Registration**: Aligning images with radiometric distortions
- **Visual Localization**: Matching query images against a database under varying conditions

## Performance Characteristics

- **Radiometric Robustness**: Handles gamma corrections, intensity shifts, and contrast changes
- **Geometric Robustness**: Invariant to rotation, scale, and limited affine transformations
- **Spectral Robustness**: Works across different spectral bands and imaging modalities
- **Computational Efficiency**: Optimized for practical applications

## Citation

If you use BIFT in your research, please cite:

```
@article{bift2024,
  title={Robust Multimodal Image Matching via Biological-inspired Invariant Feature Transform},
  author={BIFT Contributors},
  journal={Remote Sensing},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

BIFT is inspired by biological vision systems and builds upon classical feature detection methods like SIFT and SURF, while specifically addressing challenges in multimodal remote sensing image matching.
