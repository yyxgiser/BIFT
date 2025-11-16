"""
BIFT command-line interface.

Run BIFT directly: python -m bift --help
"""

import sys
import argparse
import cv2
import numpy as np
from . import BIFT


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='BIFT: Biological-inspired Invariant Feature Transform for Image Matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect keypoints in an image
  python -m bift detect image.png --output keypoints.png
  
  # Match two images
  python -m bift match image1.png image2.png --output matches.png
  
  # Show version
  python -m bift --version
        """
    )
    
    parser.add_argument('--version', action='version', version='BIFT 0.1.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect keypoints in an image')
    detect_parser.add_argument('image', help='Input image path')
    detect_parser.add_argument('--output', '-o', help='Output visualization path')
    detect_parser.add_argument('--max-keypoints', type=int, default=1000,
                              help='Maximum keypoints to detect (default: 1000)')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match features between two images')
    match_parser.add_argument('image1', help='First image path')
    match_parser.add_argument('image2', help='Second image path')
    match_parser.add_argument('--output', '-o', help='Output visualization path')
    match_parser.add_argument('--max-matches', type=int, default=50,
                             help='Maximum matches to display (default: 50)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Initialize BIFT
    bift = BIFT()
    
    if args.command == 'detect':
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image '{args.image}'")
            return 1
        
        # Detect keypoints
        print(f"Detecting keypoints in '{args.image}'...")
        keypoints, descriptors = bift.detectAndCompute(image)
        print(f"Detected {len(keypoints)} keypoints")
        
        # Visualize if output specified
        if args.output:
            vis = bift.visualize_keypoints(image, keypoints, 
                                          max_keypoints=args.max_keypoints)
            cv2.imwrite(args.output, vis)
            print(f"Saved visualization to '{args.output}'")
    
    elif args.command == 'match':
        # Load images
        image1 = cv2.imread(args.image1)
        image2 = cv2.imread(args.image2)
        
        if image1 is None:
            print(f"Error: Could not load image '{args.image1}'")
            return 1
        if image2 is None:
            print(f"Error: Could not load image '{args.image2}'")
            return 1
        
        # Match features
        print(f"Matching features between '{args.image1}' and '{args.image2}'...")
        kp1, kp2, matches, H = bift.match(image1, image2)
        
        print(f"Detected {len(kp1)} keypoints in image 1")
        print(f"Detected {len(kp2)} keypoints in image 2")
        print(f"Found {len(matches)} matches")
        
        if H is not None:
            print("Homography estimated successfully")
        
        # Visualize if output specified
        if args.output:
            vis = bift.visualize_matches(image1, image2, kp1, kp2, matches,
                                        max_matches=args.max_matches)
            cv2.imwrite(args.output, vis)
            print(f"Saved visualization to '{args.output}'")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
