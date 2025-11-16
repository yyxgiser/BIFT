"""
BIFT: Biological-inspired Invariant Feature Transform
For Robust Multimodal Image Matching
"""

from .detector import BIFTDetector
from .descriptor import BIFTDescriptor
from .matcher import BIFTMatcher
from .bift import BIFT

__version__ = "0.1.0"
__all__ = ['BIFT', 'BIFTDetector', 'BIFTDescriptor', 'BIFTMatcher']
