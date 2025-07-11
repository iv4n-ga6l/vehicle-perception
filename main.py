import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any

from lane_detector import LaneDetector
from vehicle_detector import VehicleDetector
from bev_processor import BirdEyeViewProcessor
from overlay_manager import OverlayManager

from pycache_handler.handler import py_cache_handler



class VehiclePerceptionSystem:
    """Main system orchestrating all perception components"""
    
    def __init__(self, video_path: str, output_path: Optional[str] = None):
        self.video_path = video_path
        self.output_path = output_path
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Initialize components
        self.lane_detector = LaneDetector()
        self.vehicle_detector = VehicleDetector()
        self.bev_processor = BirdEyeViewProcessor((self.frame_height, self.frame_width))
        self.overlay_manager = OverlayManager((self.frame_height, self.frame_width))
        
        # Initialize video writer if output path provided
        self.video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                               (self.frame_width, self.frame_height))
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame through all perception components"""
        frame_start_time = time.time()
        
        # 1. Lane Detection
        left_lane, right_lane = self.lane_detector.detect_lane_lines(frame)
        self.lane_detector.draw_lanes(frame, left_lane, right_lane)
        
        # 2. Vehicle Detection
        detections = self.vehicle_detector.detect_vehicles(frame)
        vehicle_frame = self.vehicle_detector.draw_detections(frame, detections)
        
        # 3. Create 3D visualization for vehicle detection
        viz_3d = self.vehicle_detector.create_3d_visualization(detections, frame.shape)
        
        # 4. Combine lane detection with 3D visualization
        combined_3d_lanes = self.lane_detector.draw_lanes(viz_3d, left_lane, right_lane)
        
        # 5. Bird's Eye View with Depth
        bev_frame = self.bev_processor.process_frame(frame)
        
        # 6. Create quadrant display with separated components
        final_frame = self.overlay_manager.create_quadrant_display(
            frame,                # Original video
            vehicle_frame,        # Vehicle detection with 3D bounding boxes only
            bev_frame,           # Bird's eye view + depth
            combined_3d_lanes,   # 3D vehicle positions + lane detection
            detections,          # Detection info
            {
                # 'Frame': self.frame_count,
                # 'Lanes': 'Detected' if (left_lane is not None or right_lane is not None) else 'None'
            }
        )
        
        # 7. Add FPS counter
        processing_time = time.time() - frame_start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        final_frame = self.overlay_manager.add_fps_counter(final_frame, current_fps)
        
        # Prepare frame info
        frame_info = {
            'detections': detections,
            'lane_detected': left_lane is not None or right_lane is not None,
            'processing_time': processing_time,
            'fps': current_fps
        }
        
        return final_frame, frame_info
    
    def run(self, display: bool = True, save_output: bool = True) -> None:
        """Run the vehicle perception system"""
        print("Starting vehicle perception system...")
        print(f"Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, frame_info = self.process_frame(frame)
                
                # Save frame if video writer is available
                if self.video_writer and save_output:
                    self.video_writer.write(processed_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Vehicle Perception System', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = f'frame_{self.frame_count:06d}.jpg'
                        cv2.imwrite(save_path, processed_frame)
                        print(f"Saved frame to {save_path}")
                
                self.frame_count += 1
                
                # Print progress
                if self.frame_count % 30 == 0:  # Every 30 frames
                    progress = (self.frame_count / self.total_frames) * 100
                    elapsed_time = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed_time
                    print(f"Progress: {progress:.1f}% | Frame: {self.frame_count}/{self.total_frames} | "
                          f"Avg FPS: {avg_fps:.1f} | Detections: {len(frame_info['detections'])}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            self.cleanup()
    
    def process_single_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Process a single frame by frame number"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        processed_frame, _ = self.process_frame(frame)
        return processed_frame
    
    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """Get and process frame at specific time"""
        frame_number = int(time_seconds * self.fps)
        return self.process_single_frame(frame_number)
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {self.frame_count} frames")
        if self.frame_count > 0:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time
            print(f"Average processing speed: {avg_fps:.2f} FPS")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@py_cache_handler
def main():
    video_path = "source.mp4"
    output_path = "output_processed.mp4"
    
    try:
        with VehiclePerceptionSystem(video_path, output_path) as system:
            system.run(display=True, save_output=True)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
