import cv2
import numpy as np
from typing import List, Tuple, Optional


class LaneDetector:
    """Lane detection using computer vision techniques"""
    
    def __init__(self):
        self.roi_vertices = None
        self.prev_left_fit = None
        self.prev_right_fit = None
        
    def region_of_interest(self, img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Apply region of interest mask to image"""
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def detect_edges(self, img: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges
    
    def detect_lane_lines(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect lane lines using Hough transform"""
        height, width = img.shape[:2]
        
        # Define region of interest
        if self.roi_vertices is None:
            self.roi_vertices = np.array([
                [(0, height),
                 (width // 2 - 50, height // 2 + 50),
                 (width // 2 + 50, height // 2 + 50),
                 (width, height)]
            ], dtype=np.int32)
        
        # Edge detection
        edges = self.detect_edges(img)
        roi_edges = self.region_of_interest(edges, self.roi_vertices)
        
        # Hough line detection
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
        
        if lines is None:
            return None, None
        
        # Separate left and right lines
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Vertical line
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.5:  # Left lane
                left_lines.append(line[0])
            elif slope > 0.5:  # Right lane
                right_lines.append(line[0])
        
        # Average lines
        left_lane = self._average_lines(left_lines) if left_lines else None
        right_lane = self._average_lines(right_lines) if right_lines else None
        
        return left_lane, right_lane
    
    def _average_lines(self, lines: List[np.ndarray]) -> np.ndarray:
        """Average multiple lines into a single line"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        # Fit line
        poly = np.polyfit(y_coords, x_coords, 1)
        
        # Calculate start and end points
        y1 = int(max(y_coords))
        y2 = int(min(y_coords))
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])
        
        return np.array([x1, y1, x2, y2])
    
    def draw_lanes(self, img: np.ndarray, left_lane: np.ndarray, right_lane: np.ndarray) -> np.ndarray:
        """Draw detected lanes on image"""
        line_img = np.zeros_like(img)
        
        if left_lane is not None:
            cv2.line(line_img, (left_lane[0], left_lane[1]), 
                    (left_lane[2], left_lane[3]), (0, 255, 0), 10)
        
        if right_lane is not None:
            cv2.line(line_img, (right_lane[0], right_lane[1]), 
                    (right_lane[2], right_lane[3]), (0, 255, 0), 10)
        
        return cv2.addWeighted(img, 0.8, line_img, 1, 0)
