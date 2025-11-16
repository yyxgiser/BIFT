# BIFT Examples

This directory contains example scripts demonstrating various uses of the BIFT algorithm.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the simplest way to use BIFT for feature detection and matching.

```bash
python basic_usage.py
```

**What it shows:**
- Creating and initializing BIFT
- Detecting keypoints and computing descriptors
- Matching features between images
- Basic visualization

### 2. Demo (`demo.py`)

Complete demonstration of BIFT capabilities with synthetic multimodal images.

```bash
python demo.py
```

**What it shows:**
- Creating synthetic multimodal images with radiometric distortions
- Full BIFT pipeline: detection, description, and matching
- Handling rotation, scale, gamma correction, and noise
- RANSAC-based outlier filtering
- Comprehensive visualization and statistics

## Output

All examples save their results to an `output/` directory (created automatically).

## Requirements

Make sure you have installed BIFT and its dependencies:

```bash
pip install -r ../requirements.txt
```

## Customization

You can customize BIFT parameters in any example. Key parameters include:

```python
bift = BIFT(
    num_octaves=4,           # Number of scale octaves
    scales_per_octave=5,     # Scales per octave
    contrast_threshold=0.04, # Keypoint contrast threshold
    max_keypoints=10000,     # Maximum keypoints to detect
    ratio_threshold=0.8      # Matching ratio test threshold
)
```

## Using Your Own Images

To use BIFT with your own images:

```python
import cv2
from bift import BIFT

# Load your images
image1 = cv2.imread('your_image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('your_image2.png', cv2.IMREAD_GRAYSCALE)

# Initialize and match
bift = BIFT()
kp1, kp2, matches, H = bift.match(image1, image2)

# Visualize
result = bift.visualize_matches(image1, image2, kp1, kp2, matches)
cv2.imwrite('result.png', result)
```
