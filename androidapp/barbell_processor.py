"""
Barbell detection and tracking processor module.
Wraps the YOLO-based detection logic for use in the Flet app.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import cv2
import numpy as np
from scipy.signal import savgol_filter

# Add parent directory to import YOLO utilities
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO


@dataclass
class VideoProcessingResult:
    """Contains all results from video processing"""
    output_video_path: str
    repetition_count: int
    total_frames: int
    averaged_path: Optional[List[Tuple[float, float]]]
    averaged_down_phase: Optional[Tuple[np.ndarray, np.ndarray]]  # (x_coords, y_coords)
    averaged_up_phase: Optional[Tuple[np.ndarray, np.ndarray]]    # (x_coords, y_coords)
    all_cycles: List[List[Tuple[int, int]]]
    video_dimensions: Tuple[int, int]  # (width, height)
    success: bool
    error_message: Optional[str] = None


class TrackingState:
    """Maintains state for barbell tracking across frames"""
    
    def __init__(self):
        self.centers: List[Tuple[int, int]] = []
        self.all_cycles: List[List[Tuple[int, int]]] = []
        self.current_cycle: List[Tuple[int, int]] = []
        self.trail_points: List[dict] = []
        self.bbox_history: List[np.ndarray] = []
        self.is_going_down: Optional[bool] = None


class BarbellPathDetector:
    """Handles barbell detection and path tracking"""
    
    # Configuration constants
    MIN_CYCLE_LENGTH = 30
    FADE_DURATION = 60
    BBOX_SMOOTH_WINDOW = 5
    DETECTION_CONF = 0.25
    DETECTION_IOU = 0.45
    
    def __init__(self, model_path: str):
        """Initialize the detector with a YOLO model"""
        self.model = YOLO(model_path)
        self._setup_device()
    
    def _setup_device(self):
        """Configure CUDA or CPU for inference"""
        try:
            self.model.to("cuda")
        except:
            self.model.to("cpu")
    
    @staticmethod
    def smooth(arr: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to an array"""
        if len(arr) < 7:
            return arr
        
        window_length = min(11, len(arr) - (1 - len(arr) % 2))
        return savgol_filter(arr, window_length, 3, mode="interp")
    
    def detect_and_track(
        self,
        video_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        rotate: int = 0
    ) -> VideoProcessingResult:
        """
        Process video to detect and track barbell movement
        
        Args:
            video_path: Path to input video
            output_path: Path for output video
            progress_callback: Optional callback for progress updates (progress, message)
            rotate: Rotation angle (0, 90, -90)
        
        Returns:
            VideoProcessingResult with all processing data
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_error_result("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = self._get_video_dimensions(cap, rotate)
            
            # Create video writer
            out = self._create_video_writer(output_path, fps, width, height)
            
            # Initialize tracking state
            tracker = TrackingState()
            frame_idx = 0
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self._apply_rotation(frame, rotate)
                frame = self._process_frame(frame, tracker)
                
                out.write(frame)
                frame_idx += 1
                
                # Update progress
                if progress_callback and frame_idx % 10 == 0:
                    progress = frame_idx / total_frames_count
                    progress_callback(progress, f"Processing frame {frame_idx}/{total_frames_count}")
            
            # Finalize
            cap.release()
            out.release()
            
            if progress_callback:
                progress_callback(1.0, "Processing complete!")
            
            # Add last cycle if valid
            if len(tracker.current_cycle) > self.MIN_CYCLE_LENGTH:
                tracker.all_cycles.append(tracker.current_cycle)
            
            # Compute averaged paths
            averaged_path, avg_down, avg_up = self._compute_averaged_path(tracker.all_cycles)
            
            return VideoProcessingResult(
                output_video_path=output_path,
                repetition_count=len(tracker.all_cycles),
                total_frames=frame_idx,
                averaged_path=averaged_path,
                averaged_down_phase=avg_down,
                averaged_up_phase=avg_up,
                all_cycles=tracker.all_cycles,
                video_dimensions=(width, height),
                success=True
            )
            
        except Exception as e:
            return self._create_error_result(str(e))
    
    def _get_video_dimensions(self, cap, rotate: int) -> Tuple[int, int]:
        """Get video dimensions, accounting for rotation"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if abs(rotate) == 90:
            return height, width
        return width, height
    
    def _create_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """Create video writer for output"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    @staticmethod
    def _apply_rotation(frame: np.ndarray, rotate: int) -> np.ndarray:
        """Apply rotation to frame if specified"""
        if rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == -90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame
    
    def _process_frame(self, frame: np.ndarray, tracker: TrackingState) -> np.ndarray:
        """Process a single frame: detect, track, and draw"""
        # Run YOLO detection
        results = self.model(frame, conf=self.DETECTION_CONF, iou=self.DETECTION_IOU, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            center = self._process_detection(boxes, results.boxes.conf.cpu().numpy(), tracker)
            if center:
                self._detect_cycles(center, tracker)
                cv2.circle(frame, center, 4, (255, 0, 0), -1)
        
        # Draw trail
        self._draw_trail(frame, tracker)
        
        # Display rep count
        cv2.putText(
            frame,
            f"Reps: {len(tracker.all_cycles) + 1}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        
        return frame
    
    def _process_detection(
        self, 
        boxes: np.ndarray, 
        confidences: np.ndarray, 
        tracker: TrackingState
    ) -> Optional[Tuple[int, int]]:
        """Process detection and return smoothed center"""
        # Get highest confidence detection
        best_idx = np.argmax(confidences)
        tracker.bbox_history.append(boxes[best_idx])
        
        # Limit history size
        if len(tracker.bbox_history) > self.BBOX_SMOOTH_WINDOW:
            tracker.bbox_history.pop(0)
        
        # Calculate smoothed bbox
        if len(tracker.bbox_history) >= 3:
            x1, y1, x2, y2 = np.array(tracker.bbox_history).mean(axis=0)
        else:
            x1, y1, x2, y2 = tracker.bbox_history[-1]
        
        # Calculate center
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        tracker.centers.append((cx, cy))
        tracker.current_cycle.append((cx, cy))
        tracker.trail_points.append({"p": (cx, cy), "f": len(tracker.centers) - 1})
        
        return (cx, cy)
    
    def _detect_cycles(self, center: Tuple[int, int], tracker: TrackingState):
        """Detect repetition cycles based on movement direction"""
        if len(tracker.centers) < 20:
            return
        
        trend = tracker.centers[-1][1] - tracker.centers[-10][1]
        
        if trend > 10 and tracker.is_going_down != True:
            if tracker.is_going_down == False and len(tracker.current_cycle) > self.MIN_CYCLE_LENGTH:
                tracker.all_cycles.append(tracker.current_cycle.copy())
                tracker.current_cycle = [center]
            tracker.is_going_down = True
        elif trend < -10 and tracker.is_going_down != False:
            tracker.is_going_down = False
    
    def _draw_trail(self, frame: np.ndarray, tracker: TrackingState):
        """Draw the smoothed trail on the frame"""
        frame_num = len(tracker.centers)
        
        # Remove old trail points
        tracker.trail_points = [
            p for p in tracker.trail_points 
            if frame_num - p["f"] < self.FADE_DURATION
        ]
        
        if len(tracker.trail_points) < 5:
            return
        
        # Smooth trail coordinates
        xs = self.smooth(np.array([p["p"][0] for p in tracker.trail_points]))
        ys = self.smooth(np.array([p["p"][1] for p in tracker.trail_points]))
        ages = np.array([frame_num - p["f"] for p in tracker.trail_points])
        
        # Draw trail segments with fading
        for i in range(len(xs) - 1):
            alpha = 1.0 - ages[i] / self.FADE_DURATION
            
            # Green for downward, red for upward
            color = (
                (0, int(255 * alpha), 0) if ys[i + 1] > ys[i] 
                else (0, 0, int(255 * alpha))
            )
            
            cv2.line(
                frame,
                (int(xs[i]), int(ys[i])),
                (int(xs[i + 1]), int(ys[i + 1])),
                color,
                max(2, int(2 * alpha)),
                cv2.LINE_AA,
            )
    
    @staticmethod
    def _create_error_result(error_message: str) -> VideoProcessingResult:
        """Create an error result"""
        return VideoProcessingResult(
            output_video_path="",
            repetition_count=0,
            total_frames=0,
            averaged_path=None,
            averaged_down_phase=None,
            averaged_up_phase=None,
            all_cycles=[],
            video_dimensions=(0, 0),
            success=False,
            error_message=error_message
        )
    
    def _compute_averaged_path(
        self, all_cycles: List[List[Tuple[int, int]]]
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute the averaged bar path from all cycles
        
        Returns:
            Tuple of (combined_path, down_phase, up_phase)
        """
        if not all_cycles:
            return None, None, None
        
        down_phases, up_phases = self._split_cycles(all_cycles)
        
        avg_down = self._average_phase(down_phases)
        avg_up = self._average_phase(up_phases)
        
        combined = self._combine_phases(avg_down, avg_up)
        
        return combined, avg_down, avg_up
    
    @staticmethod
    def _split_cycles(all_cycles: List[List[Tuple[int, int]]]) -> Tuple[List, List]:
        """Split cycles into downward and upward phases"""
        down_phases = []
        up_phases = []
        
        for cycle in all_cycles:
            if len(cycle) < 10:
                continue
            
            # Find lowest point (bottom of movement)
            lowest = np.argmax([p[1] for p in cycle])
            
            if 5 < lowest < len(cycle) - 5:
                down_phases.append(cycle[: lowest + 1])
                up_phases.append(cycle[lowest:])
        
        return down_phases, up_phases
    
    def _average_phase(self, phases: List) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Average a list of movement phases"""
        if not phases:
            return None
        
        max_len = max(len(p) for p in phases)
        normalized = []
        
        for phase in phases:
            xs = np.interp(
                np.linspace(0, len(phase) - 1, max_len),
                range(len(phase)),
                [pt[0] for pt in phase]
            )
            ys = np.interp(
                np.linspace(0, len(phase) - 1, max_len),
                range(len(phase)),
                [pt[1] for pt in phase]
            )
            normalized.append((xs, ys))
        
        avg_x = np.mean([n[0] for n in normalized], axis=0)
        avg_y = np.mean([n[1] for n in normalized], axis=0)
        
        return self.smooth(avg_x), self.smooth(avg_y)
    
    @staticmethod
    def _combine_phases(
        down_phase: Optional[Tuple[np.ndarray, np.ndarray]], 
        up_phase: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[List[Tuple[float, float]]]:
        """Combine down and up phases into a single path"""
        if not down_phase or not up_phase:
            return None
        
        combined = []
        
        for i in range(len(down_phase[0])):
            combined.append((float(down_phase[0][i]), float(down_phase[1][i])))
        
        for i in range(len(up_phase[0])):
            combined.append((float(up_phase[0][i]), float(up_phase[1][i])))
        
        return combined


def process_video(
    video_path: str,
    model_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    rotate: int = 0
) -> VideoProcessingResult:
    """
    Main entry point for video processing
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLO model file
        output_dir: Directory for output files
        progress_callback: Optional callback for progress updates
        rotate: Rotation angle (0, 90, -90)
    
    Returns:
        VideoProcessingResult with all processing data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    input_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_processed.mp4")
    
    # Create detector and process
    detector = BarbellPathDetector(model_path)
    result = detector.detect_and_track(
        video_path, output_path, progress_callback, rotate
    )
    
    return result
