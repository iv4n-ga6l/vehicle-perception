import cv2
import numpy as np
from typing import Tuple, Optional


class BirdEyeViewProcessor:
    """Bird's eye view transformation and depth estimation"""
    
    def __init__(self, img_shape: Tuple[int, int]):
        self.img_height, self.img_width = img_shape[:2]
        self.bev_width = 400
        self.bev_height = 600
        
        # Define source and destination points for perspective transformation
        self.src_points = self._get_source_points()
        self.dst_points = self._get_destination_points()
        
        # Get transformation matrices
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        
        # Depth estimation parameters
        self.depth_scale = 255.0 / 50.0  # Assume max depth of 50 meters
        
    def _get_source_points(self) -> np.ndarray:
        """Define source points for perspective transformation"""
        # These points should form a trapezoid representing the road ahead
        return np.float32([
            [self.img_width * 0.1, self.img_height * 0.95],      # bottom-left
            [self.img_width * 0.9, self.img_height * 0.95],      # bottom-right
            [self.img_width * 0.6, self.img_height * 0.6],       # top-right
            [self.img_width * 0.4, self.img_height * 0.6]        # top-left
        ])
    
    def _get_destination_points(self) -> np.ndarray:
        """Define destination points for bird's eye view"""
        return np.float32([
            [50, self.bev_height - 50],                          # bottom-left
            [self.bev_width - 50, self.bev_height - 50],         # bottom-right
            [self.bev_width - 50, 50],                           # top-right
            [50, 50]                                             # top-left
        ])
    
    def transform_to_bev(self, img: np.ndarray) -> np.ndarray:
        """Transform image to bird's eye view"""
        bev_img = cv2.warpPerspective(img, self.M, (self.bev_width, self.bev_height))
        return bev_img
    
    def transform_from_bev(self, bev_img: np.ndarray) -> np.ndarray:
        """Transform bird's eye view back to original perspective"""
        return cv2.warpPerspective(bev_img, self.M_inv, (self.img_width, self.img_height))
    
    def estimate_depth(self, img: np.ndarray) -> np.ndarray:
        """Estimate depth map using simple computer vision techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use gradient-based depth estimation
        # Objects closer to bottom of image are assumed to be closer
        height, width = gray.shape
        
        # Create depth map based on vertical position
        depth_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(height):
            # Linear depth based on vertical position
            depth_value = (height - y) / height
            depth_map[y, :] = depth_value
        
        # Enhance depth using edges (edges often indicate depth discontinuities)
        edges = cv2.Canny(gray, 50, 150)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # Combine positional depth with edge information
        depth_map = 0.7 * depth_map + 0.3 * edges_normalized
        
        # Apply some smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # Convert to 8-bit for visualization
        depth_map_8bit = (depth_map * 255).astype(np.uint8)
        
        return depth_map_8bit
    
    def create_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """Create colored depth map for visualization"""
        # Apply color map (closer objects are warmer colors)
        colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        return colored_depth
    
    def create_bev_with_depth(self, img: np.ndarray) -> np.ndarray:
        """Create bird's eye view with depth information"""
        # Transform to bird's eye view
        bev_img = self.transform_to_bev(img)
        
        # Estimate depth
        depth_map = self.estimate_depth(bev_img)
        depth_colored = self.create_depth_colormap(depth_map)
        
        # Blend original BEV with depth information
        result = cv2.addWeighted(bev_img, 0.6, depth_colored, 0.4, 0)
        
        return result
    
    def draw_grid(self, img: np.ndarray, grid_size: int = 50) -> np.ndarray:
        """Draw grid lines on bird's eye view for scale reference"""
        result = img.copy()
        height, width = img.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, grid_size):
            cv2.line(result, (x, 0), (x, height), (100, 100, 100), 1)
        
        # Draw horizontal lines
        for y in range(0, height, grid_size):
            cv2.line(result, (0, y), (width, y), (100, 100, 100), 1)
        
        # Add distance markers
        for y in range(grid_size, height, grid_size):
            distance = (height - y) * 0.1  # Approximate distance in meters
            cv2.putText(result, f"{distance:.1f}m", (10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return result
    
    def process_frame(self, img: np.ndarray) -> np.ndarray:
        """Process a single frame to create BEV with depth"""
        # Create BEV with depth
        bev_depth = self.create_bev_with_depth(img)
        
        # Add grid for reference
        bev_with_grid = self.draw_grid(bev_depth)
        
        return bev_with_grid
