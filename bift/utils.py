"""
Utility functions for BIFT algorithm
"""

import numpy as np
import cv2
from typing import Tuple, List


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to handle radiometric distortions.
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Normalized image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to [0, 1] range
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    
    return image


def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and orientation.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Tuple of (magnitude, orientation) arrays
    """
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x)
    
    return magnitude, orientation


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel.
    
    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation
        
    Returns:
        Gaussian kernel
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def build_scale_space(image: np.ndarray, 
                      num_octaves: int = 4, 
                      scales_per_octave: int = 5,
                      sigma: float = 1.6) -> List[List[np.ndarray]]:
    """
    Build scale space pyramid for biological-inspired feature detection.
    
    Args:
        image: Input grayscale image
        num_octaves: Number of octaves in the pyramid
        scales_per_octave: Number of scales per octave
        sigma: Initial sigma for Gaussian blur
        
    Returns:
        List of octaves, each containing a list of blurred images
    """
    k = 2 ** (1.0 / scales_per_octave)
    scale_space = []
    
    for octave in range(num_octaves):
        octave_images = []
        # Downsample image for this octave
        if octave == 0:
            base_image = image
        else:
            base_image = cv2.resize(scale_space[octave-1][-3], 
                                   (image.shape[1] // (2**octave), 
                                    image.shape[0] // (2**octave)),
                                   interpolation=cv2.INTER_LINEAR)
        
        # Generate scales for this octave
        for scale_idx in range(scales_per_octave + 3):
            sigma_scale = sigma * (k ** scale_idx)
            ksize = int(2 * np.ceil(3 * sigma_scale) + 1)
            blurred = cv2.GaussianBlur(base_image, (ksize, ksize), sigma_scale)
            octave_images.append(blurred)
        
        scale_space.append(octave_images)
    
    return scale_space


def compute_dog_pyramid(scale_space: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """
    Compute Difference of Gaussian (DoG) pyramid from scale space.
    Biological-inspired edge detection similar to retinal processing.
    
    Args:
        scale_space: Scale space pyramid
        
    Returns:
        DoG pyramid
    """
    dog_pyramid = []
    
    for octave_images in scale_space:
        octave_dogs = []
        for i in range(len(octave_images) - 1):
            dog = octave_images[i+1] - octave_images[i]
            octave_dogs.append(dog)
        dog_pyramid.append(octave_dogs)
    
    return dog_pyramid


def is_valid_keypoint(x: int, y: int, image_shape: Tuple[int, int], border: int = 5) -> bool:
    """
    Check if keypoint is valid (not too close to border).
    
    Args:
        x: X coordinate
        y: Y coordinate
        image_shape: Shape of the image (height, width)
        border: Minimum distance from border
        
    Returns:
        True if keypoint is valid
    """
    height, width = image_shape
    return (border <= x < width - border and 
            border <= y < height - border)


def subpixel_refinement(dog_pyramid: List[List[np.ndarray]], 
                       octave: int, 
                       scale: int, 
                       y: int, 
                       x: int) -> Tuple[float, float, float]:
    """
    Refine keypoint location to subpixel accuracy using quadratic interpolation.
    
    Args:
        dog_pyramid: DoG pyramid
        octave: Octave index
        scale: Scale index
        y: Y coordinate
        x: X coordinate
        
    Returns:
        Refined (x, y, scale) coordinates
    """
    # Get 3x3x3 cube around the point
    if (scale <= 0 or scale >= len(dog_pyramid[octave]) - 1 or
        y <= 0 or y >= dog_pyramid[octave][scale].shape[0] - 1 or
        x <= 0 or x >= dog_pyramid[octave][scale].shape[1] - 1):
        return x, y, scale
    
    # Compute gradient and Hessian
    cube = np.array([
        dog_pyramid[octave][scale-1][y-1:y+2, x-1:x+2],
        dog_pyramid[octave][scale][y-1:y+2, x-1:x+2],
        dog_pyramid[octave][scale+1][y-1:y+2, x-1:x+2]
    ])
    
    # Compute derivatives
    dx = (cube[1, 1, 2] - cube[1, 1, 0]) / 2.0
    dy = (cube[1, 2, 1] - cube[1, 0, 1]) / 2.0
    ds = (cube[2, 1, 1] - cube[0, 1, 1]) / 2.0
    
    gradient = np.array([dx, dy, ds])
    
    # Compute Hessian
    dxx = cube[1, 1, 2] - 2 * cube[1, 1, 1] + cube[1, 1, 0]
    dyy = cube[1, 2, 1] - 2 * cube[1, 1, 1] + cube[1, 0, 1]
    dss = cube[2, 1, 1] - 2 * cube[1, 1, 1] + cube[0, 1, 1]
    dxy = ((cube[1, 2, 2] - cube[1, 2, 0]) - (cube[1, 0, 2] - cube[1, 0, 0])) / 4.0
    dxs = ((cube[2, 1, 2] - cube[2, 1, 0]) - (cube[0, 1, 2] - cube[0, 1, 0])) / 4.0
    dys = ((cube[2, 2, 1] - cube[2, 0, 1]) - (cube[0, 2, 1] - cube[0, 0, 1])) / 4.0
    
    hessian = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ])
    
    # Solve for offset
    try:
        offset = -np.linalg.solve(hessian, gradient)
        
        # Check if offset is reasonable
        if np.abs(offset).max() > 1.5:
            return x, y, scale
        
        return x + offset[0], y + offset[1], scale + offset[2]
    except np.linalg.LinAlgError:
        return x, y, scale
