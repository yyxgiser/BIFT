# Changelog

All notable changes to the BIFT project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-16

### Added
- Initial implementation of BIFT (Biological-inspired Invariant Feature Transform) algorithm
- BIFTDetector: Scale-invariant keypoint detection using DoG pyramid
- BIFTDescriptor: 128-dimensional rotation and illumination-invariant descriptors
- BIFTMatcher: Robust feature matching with ratio test and RANSAC filtering
- Complete BIFT pipeline with visualization utilities
- Utility functions for:
  - Image normalization for radiometric invariance
  - Scale space and DoG pyramid construction
  - Gradient computation
  - Subpixel keypoint refinement
- Comprehensive documentation and README
- Example scripts:
  - Full demo with synthetic multimodal images
  - Basic usage example
  - Parameter tuning example
- Test suite with 5 test cases
- MIT License
- Contributing guidelines

### Features
- Scale invariance through multi-scale DoG detection
- Rotation invariance through dominant orientation assignment
- Radiometric invariance through normalization and descriptor design
- Support for multimodal remote sensing image matching
- Handles nonlinear radiometric distortions
- Robust to spectral discrepancies
- RANSAC-based homography estimation for outlier filtering
- Visualization tools for keypoints and matches

### Performance
- Efficient scale-space construction
- Optimized descriptor computation
- Fast matching with ratio test
- Configurable parameters for speed/accuracy tradeoff

## [Unreleased]

### Planned
- GPU acceleration support
- Additional descriptor variants
- More matching strategies
- Performance benchmarks
- Additional examples with real remote sensing data
