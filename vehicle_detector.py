import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any


class VehicleDetector:
    """Vehicle detection using YOLOv8"""
    
    def __init__(self, model_path: str = None):
        """Initialize YOLOv8 model"""
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Use pre-trained YOLOv8 model
            self.model = YOLO('yolo11n.pt')
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Colors for different vehicle types
        self.colors = {
            2: (0, 255, 0),    # car - green
            3: (255, 0, 0),    # motorcycle - blue
            5: (0, 0, 255),    # bus - red
            7: (255, 255, 0)   # truck - cyan
        }
    
    def detect_vehicles(self, img: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Detect vehicles in image using YOLOv8"""
        results = self.model(img, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for vehicles only
                    if class_id in self.vehicle_classes and confidence >= confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': self.class_names[class_id],
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                        detections.append(detection)
        
        return detections
    
    def estimate_3d_position(self, detection: Dict[str, Any], img_shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate 3D position of vehicle (simplified approach)"""
        height, width = img_shape[:2]
        
        # Simple depth estimation based on object size and position
        # This is a simplified approach - in real applications, you'd use stereo vision or LiDAR
        bbox = detection['bbox']
        object_width = bbox[2] - bbox[0]
        object_height = bbox[3] - bbox[1]
        
        # Estimated focal length for depth calculations
        focal_length = 800  # estimated focal length
        
        # Estimate distance based on object size (larger objects are closer)
        # This is a rough approximation
        if detection['class_name'] == 'car':
            typical_car_width = 1.8  # meters
            distance = (typical_car_width * focal_length) / object_width
        elif detection['class_name'] == 'truck':
            typical_truck_width = 2.5  # meters
            distance = (typical_truck_width * focal_length) / object_width
        elif detection['class_name'] == 'bus':
            typical_bus_width = 2.6  # meters
            distance = (typical_bus_width * focal_length) / object_width
        elif detection['class_name'] == 'motorcycle':
            typical_motorcycle_width = 0.8  # meters
            distance = (typical_motorcycle_width * focal_length) / object_width
        else:
            distance = 10.0  # default distance
        
        # Estimate lateral position
        center_x = detection['center'][0]
        lateral_offset = (center_x - width // 2) * distance / focal_length
        
        return {
            'distance': distance,
            'lateral_offset': lateral_offset,
            'height': 0.0  # assuming ground level
        }
    
    def draw_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw vehicle detections with 3D bounding boxes"""
        result_img = img.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = self.colors[detection['class_id']]
            
            # Draw 2D bounding box
            cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw 3D bounding box effect
            self._draw_3d_box(result_img, bbox, color, detection)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(result_img, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance estimation
            pos_3d = self.estimate_3d_position(detection, img.shape)
            distance_text = f"{pos_3d['distance']:.1f}m"
            cv2.putText(result_img, distance_text, (bbox[0], bbox[3] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center point
            center = detection['center']
            cv2.circle(result_img, center, 5, color, -1)
        
        return result_img
    
    def _draw_3d_box(self, img: np.ndarray, bbox: List[int], color: Tuple[int, int, int], detection: Dict[str, Any]):
        """Draw 3D bounding box effect"""
        x1, y1, x2, y2 = bbox
        
        # Calculate 3D box offset based on distance (closer objects have larger offset)
        pos_3d = self.estimate_3d_position(detection, img.shape)
        distance = pos_3d['distance']
        
        # Scale offset inversely with distance (closer = more 3D effect)
        offset = max(5, int(30 / max(distance, 1)))
        
        # Draw 3D effect lines
        # Top face
        cv2.line(img, (x1, y1), (x1 - offset, y1 - offset), color, 2)
        cv2.line(img, (x2, y1), (x2 - offset, y1 - offset), color, 2)
        cv2.line(img, (x1 - offset, y1 - offset), (x2 - offset, y1 - offset), color, 2)
        
        # Side face
        cv2.line(img, (x2, y1), (x2 - offset, y1 - offset), color, 2)
        cv2.line(img, (x2, y2), (x2 - offset, y2 - offset), color, 2)
        cv2.line(img, (x2 - offset, y1 - offset), (x2 - offset, y2 - offset), color, 2)
    def draw_detections(self, img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw vehicle detections with 3D bounding boxes"""
        result_img = img.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            color = self.colors[detection['class_id']]
            
            # Draw 2D bounding box
            cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw 3D bounding box effect
            self._draw_3d_box(result_img, bbox, color, detection)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(result_img, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw distance estimation
            pos_3d = self.estimate_3d_position(detection, img.shape)
            distance_text = f"{pos_3d['distance']:.1f}m"
            cv2.putText(result_img, distance_text, (bbox[0], bbox[3] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center point
            center = detection['center']
            cv2.circle(result_img, center, 5, color, -1)
        
        return result_img
    
    def _draw_3d_box(self, img: np.ndarray, bbox: List[int], color: Tuple[int, int, int], detection: Dict[str, Any]):
        """Draw 3D bounding box effect"""
        x1, y1, x2, y2 = bbox
        
        # Calculate 3D box offset based on distance (closer objects have larger offset)
        pos_3d = self.estimate_3d_position(detection, img.shape)
        distance = pos_3d['distance']
        
        # Scale offset inversely with distance (closer = more 3D effect)
        offset = max(5, int(30 / max(distance, 1)))
        
        # Draw 3D effect lines
        # Top face
        cv2.line(img, (x1, y1), (x1 - offset, y1 - offset), color, 2)
        cv2.line(img, (x2, y1), (x2 - offset, y1 - offset), color, 2)
        cv2.line(img, (x1 - offset, y1 - offset), (x2 - offset, y1 - offset), color, 2)
        
        # Side face
        cv2.line(img, (x2, y1), (x2 - offset, y1 - offset), color, 2)
        cv2.line(img, (x2, y2), (x2 - offset, y2 - offset), color, 2)
        cv2.line(img, (x2 - offset, y1 - offset), (x2 - offset, y2 - offset), color, 2)
        
        # Connect back edges
        cv2.line(img, (x1 - offset, y1 - offset), (x1 - offset, y2 - offset), color, 1)
        cv2.line(img, (x1 - offset, y2 - offset), (x2 - offset, y2 - offset), color, 1)
    
    def create_3d_visualization(self, detections: List[Dict[str, Any]], img_shape: Tuple[int, int]) -> np.ndarray:
        """Create a top-down 3D visualization of detected vehicles"""
        viz_height, viz_width = 300, 400
        viz_img = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
        
        # Draw coordinate system
        cv2.line(viz_img, (viz_width // 2, 0), (viz_width // 2, viz_height), (100, 100, 100), 1)
        cv2.line(viz_img, (0, viz_height - 50), (viz_width, viz_height - 50), (100, 100, 100), 1)
        
        # Draw ego vehicle
        ego_x, ego_y = viz_width // 2, viz_height - 30
        cv2.rectangle(viz_img, (ego_x - 10, ego_y - 15), (ego_x + 10, ego_y + 15), (255, 255, 255), -1)
        cv2.putText(viz_img, "EGO", (ego_x - 15, ego_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw detected vehicles
        for detection in detections:
            pos_3d = self.estimate_3d_position(detection, img_shape)
            
            # Convert 3D position to 2D visualization coordinates
            scale = 10  # pixels per meter
            x = int(viz_width // 2 + pos_3d['lateral_offset'] * scale)
            y = int(viz_height - 50 - pos_3d['distance'] * scale)
            
            # Keep within bounds
            x = max(5, min(viz_width - 5, x))
            y = max(5, min(viz_height - 5, y))
            
            color = self.colors[detection['class_id']]
            cv2.circle(viz_img, (x, y), 8, color, -1)
            cv2.putText(viz_img, f"{pos_3d['distance']:.1f}m", (x - 20, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return viz_img
