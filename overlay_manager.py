import cv2
import numpy as np
from typing import Tuple, Optional, Any


class OverlayManager:
    """Manages overlay positioning and rendering"""
    
    def __init__(self, main_frame_shape: Tuple[int, int]):
        self.main_height, self.main_width = main_frame_shape[:2]
        
        # Calculate quadrant dimensions
        self.quad_width = self.main_width // 2
        self.quad_height = self.main_height // 2
        self.padding = 2
        
        # Define quadrant positions
        self.top_left_pos = (0, 0)
        self.top_right_pos = (self.quad_width, 0)
        self.bottom_left_pos = (0, self.quad_height)
        self.bottom_right_pos = (self.quad_width, self.quad_height)
        
    def resize_to_quadrant(self, img: np.ndarray) -> np.ndarray:
        """Resize image to fit in quadrant"""
        target_width = self.quad_width - (self.padding * 2)
        target_height = self.quad_height - (self.padding * 2)
        return cv2.resize(img, (target_width, target_height))
    
    def add_border(self, img: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> np.ndarray:
        """Add border around overlay"""
        bordered = cv2.copyMakeBorder(img, thickness, thickness, thickness, thickness,
                                     cv2.BORDER_CONSTANT, value=color)
        return bordered
    
    def add_title(self, img: np.ndarray, title: str, position: str = "top") -> np.ndarray:
        """Add title to overlay"""
        result = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        if position == "top":
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = 25
        else:  # bottom
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = img.shape[0] - 10
        
        # Add background rectangle for better readability
        cv2.rectangle(result, (text_x - 5, text_y - 20), (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Add text
        cv2.putText(result, title, (text_x, text_y), font, font_scale, font_color, thickness)
        
        return result
    
    def place_in_quadrant(self, img: np.ndarray, quadrant: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize image and return it with its position for quadrant placement"""
        resized_img = self.resize_to_quadrant(img)
        
        if quadrant == "top_left":
            pos = (self.padding, self.padding)
        elif quadrant == "top_right":
            pos = (self.quad_width + self.padding, self.padding)
        elif quadrant == "bottom_left":
            pos = (self.padding, self.quad_height + self.padding)
        elif quadrant == "bottom_right":
            pos = (self.quad_width + self.padding, self.quad_height + self.padding)
        else:
            raise ValueError(f"Unknown quadrant: {quadrant}")
        
        return resized_img, pos
    
    def create_info_panel(self, info_dict: dict) -> np.ndarray:
        """Create an information panel with text for quadrant display"""
        # Create panel with quadrant size
        panel_width = self.quad_width - (self.padding * 2)
        panel_height = self.quad_height - (self.padding * 2)
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 2
        line_height = 30
        
        y_offset = 40
        
        # Add title
        # title = "Detection Information"
        # title_size = cv2.getTextSize(title, font, font_scale + 0.2, thickness + 1)[0]
        # title_x = (panel_width - title_size[0]) // 2
        # cv2.putText(panel, title, (title_x, y_offset), font, font_scale + 0.2, (0, 255, 255), thickness + 1)
        # y_offset += line_height + 10
        
        # Add info items
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (20, y_offset), font, font_scale, font_color, thickness)
            y_offset += line_height
            
            # Prevent text overflow
            if y_offset > panel_height - 20:
                break
        
        return panel
    
    def create_detection_summary(self, detections: list, frame_info: dict = None) -> np.ndarray:
        """Create a summary panel for detections"""
        # Count detections by type
        detection_counts = {}
        total_detections = len(detections)
        
        for detection in detections:
            class_name = detection['class_name']
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        # Create info dictionary
        info_dict = {}
        if frame_info:
            info_dict.update(frame_info)
        
        info_dict['Total Detections'] = total_detections
        
        # Add detection counts
        for vehicle_type, count in detection_counts.items():
            info_dict[f'{vehicle_type.capitalize()}s'] = count
        
        # Add detection details for first few detections
        if detections:
            # info_dict['--- Detection Details ---'] = ''
            for i, detection in enumerate(detections[:3]):  # Show first 3
                distance = self._estimate_distance(detection)
                info_dict[f'{detection["class_name"]} {i+1}'] = f'{distance:.1f}m'
        
        return self.create_info_panel(info_dict)
    
    def _estimate_distance(self, detection: dict) -> float:
        """Simple distance estimation based on bounding box size"""
        bbox_area = detection['width'] * detection['height']
        # Simple heuristic: larger bounding boxes are closer
        if bbox_area > 5000:
            return 5.0
        elif bbox_area > 2000:
            return 10.0
        elif bbox_area > 1000:
            return 20.0
        else:
            return 30.0
    
    def create_quadrant_display(self, original_frame: np.ndarray, 
                               vehicle_frame: np.ndarray, 
                               bev_frame: np.ndarray,
                               viz_3d: np.ndarray,
                               detections: list = None,
                               frame_info: dict = None) -> np.ndarray:
        """Create a 4-quadrant display with different views"""
        
        # Create the main display canvas
        display_height = self.main_height
        display_width = self.main_width
        quad_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Prepare each quadrant
        # Top-left: Original video
        orig_resized, orig_pos = self.place_in_quadrant(original_frame, "top_left")
        orig_with_title = self.add_title(orig_resized, "Base")
        
        # Top-right: Vehicle Detection with 3D bounding boxes only
        vehicle_resized, vehicle_pos = self.place_in_quadrant(vehicle_frame, "top_right")
        vehicle_with_title = self.add_title(vehicle_resized, "Detection")
        
        # Bottom-left: Bird's Eye View + Depth
        bev_resized, bev_pos = self.place_in_quadrant(bev_frame, "bottom_left")
        bev_with_title = self.add_title(bev_resized, "Bev + Depth")
        
        # Bottom-right: 3D Visualization + Lane Detection
        viz_resized, viz_pos = self.place_in_quadrant(viz_3d, "bottom_right")
        viz_with_title = self.add_title(viz_resized, "Positions")
        
        # Place all quadrants
        # Top-left quadrant
        quad_display[orig_pos[1]:orig_pos[1]+orig_with_title.shape[0], 
                    orig_pos[0]:orig_pos[0]+orig_with_title.shape[1]] = orig_with_title
        
        # Top-right quadrant
        quad_display[vehicle_pos[1]:vehicle_pos[1]+vehicle_with_title.shape[0], 
                    vehicle_pos[0]:vehicle_pos[0]+vehicle_with_title.shape[1]] = vehicle_with_title
        
        # Bottom-left quadrant
        quad_display[bev_pos[1]:bev_pos[1]+bev_with_title.shape[0], 
                    bev_pos[0]:bev_pos[0]+bev_with_title.shape[1]] = bev_with_title
        
        # Bottom-right quadrant
        quad_display[viz_pos[1]:viz_pos[1]+viz_with_title.shape[0], 
                    viz_pos[0]:viz_pos[0]+viz_with_title.shape[1]] = viz_with_title
        
        # Draw dividing lines
        # Vertical line
        cv2.line(quad_display, (self.quad_width, 0), (self.quad_width, display_height), (100, 100, 100), 2)
        # Horizontal line
        cv2.line(quad_display, (0, self.quad_height), (display_width, self.quad_height), (100, 100, 100), 2)
        
        return quad_display
    
    def add_fps_counter(self, img: np.ndarray, fps: float) -> np.ndarray:
        """Add FPS counter to image"""
        result = img.copy()
        
        fps_text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)
        thickness = 2
        
        # Position at bottom-right corner
        text_size = cv2.getTextSize(fps_text, font, font_scale, thickness)[0]
        text_x = img.shape[1] - text_size[0] - 10
        text_y = img.shape[0] - 10
        
        # Add background
        cv2.rectangle(result, (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(result, fps_text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        return result
