"""
BadmintonPoseEstimator - Core functionality for badminton pose analysis.
Adapted for Django integration from the original implementation.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
import os
from django.conf import settings


@dataclass
class FrameAnalysis:
    # Stores analysis results for a single frame
    timestamp: float
    grip_angle: float
    shoulder_angle: float
    shot_type: str
    shot_phase: Optional[str]
    within_threshold: bool
    keypoints: np.ndarray


def calculate_shuttle_distance(shuttle_box: Optional[np.ndarray], wrist_point: np.ndarray) -> float:
    # Calculate distance between shuttlecock and player's wrist
    if shuttle_box is None or wrist_point is None or len(shuttle_box) < 4:
        return float('inf')

    try:
        shuttle_center = np.array([
            (shuttle_box[0] + shuttle_box[2]) / 2,
            (shuttle_box[1] + shuttle_box[3]) / 2
        ])
        return np.linalg.norm(shuttle_center - wrist_point)
    except Exception as e:
        print(f"Distance calculation error: {str(e)}")
        return float('inf')


def is_within_threshold(grip_angle: float, shoulder_angle: float, threshold: Dict[str, float]) -> bool:
    # Check if angles are within specified thresholds
    required_keys = {'grip', 'shoulder', 'tolerance'}
    if not required_keys.issubset(threshold.keys()):
        print("Threshold dictionary is missing required keys.")
        return False

    return (
            abs(grip_angle - threshold['grip']) <= threshold['tolerance'] and
            abs(shoulder_angle - threshold['shoulder']) <= threshold['tolerance']
    )


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, connections: List[Tuple[int, int]],
                  color: Tuple[int, int, int]) -> None:
    # Draw skeletal connections on frame - optimized for valid points only
    valid_points = keypoints[:, 2] > 0.5  # Only draw points above confidence threshold

    for start_idx, end_idx in connections:
        if keypoints[start_idx, 2] > 0.5 and keypoints[end_idx, 2] > 0.5:
            try:
                start = tuple(map(int, keypoints[start_idx, :2]))
                end = tuple(map(int, keypoints[end_idx, :2]))
                cv2.line(frame, start, end, color, 2)
                cv2.circle(frame, start, 4, (0, 0, 255), -1)
                cv2.circle(frame, end, 4, (0, 0, 255), -1)
            except (IndexError, ValueError) as e:
                continue


def draw_shuttlecock(frame: np.ndarray, shuttle_box: Optional[np.ndarray]) -> None:
    # Draw shuttlecock bounding box if detected
    if shuttle_box is not None:
        try:
            cv2.rectangle(
                frame,
                (int(shuttle_box[0]), int(shuttle_box[1])),
                (int(shuttle_box[2]), int(shuttle_box[3])),
                (0, 255, 255), 2
            )
        except (IndexError, ValueError) as e:
            print(f"Shuttle drawing error: {str(e)}")


def draw_info_overlay(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    # Draw analysis information overlay
    info_lines = [
        f"Phase: {analysis.shot_phase or 'Preparing'}",
        f"Grip Angle: {analysis.grip_angle:.1f}°",
        f"Shoulder Angle: {analysis.shoulder_angle:.1f}°",
        f"Within Threshold: {'✅' if analysis.within_threshold else '❌'}"
    ]

    for i, text in enumerate(info_lines):
        try:
            y_pos = 30 + i * 30
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(
                frame,
                (8, y_pos - 25),
                (12 + text_size[0], y_pos + 5),
                (0, 0, 0),
                -1
            )
            cv2.putText(
                frame,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        except Exception as e:
            print(f"Text overlay error: {str(e)}")
            continue


class BadmintonVideoTracker:
    # Class constants for better memory efficiency
    KEYPOINTS = {
        'LEFT_SHOULDER': 5,
        'RIGHT_SHOULDER': 6,
        'LEFT_ELBOW': 7,
        'RIGHT_ELBOW': 8,
        'LEFT_WRIST': 9,
        'RIGHT_WRIST': 10
    }

    # Default connections for skeleton visualization
    DEFAULT_CONNECTIONS = [
        (5, 6),  # Left to right shoulder
        (6, 8),  # Right shoulder to right elbow
        (8, 10)  # Right elbow to right wrist
    ]

    # Phase colors for visualization (pre-computed)
    PHASE_COLORS = {
        'pre_hit': (0, 255, 0),  # Green
        'hit': (0, 0, 255),  # Red
        'follow_through': (255, 0, 0),  # Blue
        'default': (128, 128, 128)  # Gray
    }

    # Optimized angle thresholds for different shot phases
    SHOT_THRESHOLDS = {
        'smash': {
            'pre_hit': {'grip': 90, 'shoulder': 45, 'tolerance': 12},
            'hit': {'grip': 165, 'shoulder': 105, 'tolerance': 12},
            'follow_through': {'grip': 150, 'shoulder': 70, 'tolerance': 12}
        },
        'clear': {
            'pre_hit': {'grip': 150, 'shoulder': 60, 'tolerance': 12},
            'hit': {'grip': 170, 'shoulder': 70, 'tolerance': 8},
            'follow_through': {'grip': 140, 'shoulder': 50, 'tolerance': 12}
        },
        'drop_shot': {
            'pre_hit': {'grip': 95, 'shoulder': 35, 'tolerance': 12},
            'hit': {'grip': 105, 'shoulder': 45, 'tolerance': 8},
            'follow_through': {'grip': 85, 'shoulder': 25, 'tolerance': 12}
        }
    }

    def __init__(self, target_shot_type: str = 'smash', model_size: str = 'n'):
        # Modified for Django: Get model paths from settings or use default
        models_dir = getattr(settings, 'YOLO_MODELS_DIR', os.path.join(settings.MEDIA_ROOT, 'models'))
        os.makedirs(models_dir, exist_ok=True)
        
        pose_model_path = os.path.join(models_dir, f'yolo11{model_size}-pose.pt')
        shuttle_model_path = os.path.join(models_dir, 'shuttlecock.pt')
        
        # Check if models exist, if not, use default paths
        if not os.path.exists(pose_model_path):
            pose_model_path = f'yolo11{model_size}-pose.pt'
        
        if not os.path.exists(shuttle_model_path):
            shuttle_model_path = 'shuttlecock.pt'

        # Initialize YOLO models with GPU acceleration if available
        self.pose_model = YOLO(pose_model_path)
        self.shuttle_model = YOLO(shuttle_model_path)

        # Move models to GPU if available for faster inference
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.pose_model.to(device)
        self.shuttle_model.to(device)

        # Detection parameters
        self.conf_threshold = 0.6
        self.iou_threshold = 0.7

        # Tracking state
        self.target_shot_type = target_shot_type
        self.tracked_id: Optional[int] = None
        self.key_frames = {
            'pre_hit': None,
            'hit': None,
            'follow_through': None
        }

        # Cache for thresholds to avoid repeated lookups
        self._thresholds_cache = self.SHOT_THRESHOLDS.get(target_shot_type, {})

        # Modified: Store all detections for each phase
        self.phase_detections = {
            'pre_hit': [],
            'hit': [],
            'follow_through': []
        }

        # Tracking state variables
        self.last_shuttle_distance = float('inf')
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_shuttle_box: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None
        self.last_analysis: Optional[FrameAnalysis] = None

    def detect_shuttlecock(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # Consider only the top half of the frame for shuttlecock detection
        h, w = frame.shape[:2]
        top_half = frame[:h // 2, :]  # Only use top half of the frame

        # ROI-based shuttlecock detection for efficiency
        if self.prev_shuttle_box is not None:
            # Expand previous detection area by 30%
            x1, y1, x2, y2 = self.prev_shuttle_box

            # Check if previous detection was in top half, otherwise reset
            if y2 > h // 2:
                # Shuttlecock has moved to bottom half, reset to full top half search
                search_frame = top_half
            else:
                # Calculate padding with 30% expansion
                pad_x = (x2 - x1) * 0.3
                pad_y = (y2 - y1) * 0.3

                # Create ROI with padding (ensuring we stay within frame bounds)
                roi_x1 = max(0, int(x1 - pad_x))
                roi_y1 = max(0, int(y1 - pad_y))
                roi_x2 = min(w, int(x2 + pad_x))
                roi_y2 = min(h // 2, int(y2 + pad_y))  # Limit to top half

                # Only process ROI if it's a valid region
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    search_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                else:
                    search_frame = top_half
        else:
            # First detection - use entire top half
            search_frame = top_half

        # Run detection on the selected search frame
        with torch.no_grad():  # Disable gradient calculation for inference
            results = self.shuttle_model(search_frame, conf=self.conf_threshold, iou=self.iou_threshold, classes=[0])

        # Process detection results
        if results[0].boxes and len(results[0].boxes):
            confidences = results[0].boxes.conf.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best_idx = np.argmax(confidences)

            # Adjust coordinates if using ROI
            if self.prev_shuttle_box is not None and 'roi_x1' in locals():
                adjusted_box = np.array([
                    boxes[best_idx, 0] + roi_x1,
                    boxes[best_idx, 1] + roi_y1,
                    boxes[best_idx, 2] + roi_x1,
                    boxes[best_idx, 3] + roi_y1
                ])
                self.prev_shuttle_box = adjusted_box
                return adjusted_box
            else:
                # If using top half, adjust the y-coordinates
                adjusted_box = boxes[best_idx].copy()
                # No need to adjust if already using top half coordinates
                self.prev_shuttle_box = adjusted_box
                return adjusted_box

        self.prev_shuttle_box = None
        return None

    def calculate_grip_angle(self, keypoints: np.ndarray) -> float:
        # Calculate elbow joint angle between upper arm and forearm
        wrist = keypoints[self.KEYPOINTS['RIGHT_WRIST'], :2]
        elbow = keypoints[self.KEYPOINTS['RIGHT_ELBOW'], :2]
        shoulder = keypoints[self.KEYPOINTS['RIGHT_SHOULDER'], :2]

        # Create vectors from elbow to shoulder and elbow to wrist
        upper_arm = shoulder - elbow
        forearm = wrist - elbow

        # Calculate angle between vectors using dot product
        dot_product = np.dot(upper_arm, forearm)
        norm_upper = np.linalg.norm(upper_arm)
        norm_forearm = np.linalg.norm(forearm)

        if norm_upper == 0 or norm_forearm == 0:
            return 0.0

        cos_theta = np.clip(dot_product / (norm_upper * norm_forearm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def calculate_shoulder_angle(self, keypoints: np.ndarray) -> float:
        # Calculate shoulder abduction angle relative to vertical torso axis
        shoulder = keypoints[self.KEYPOINTS['RIGHT_SHOULDER'], :2]
        elbow = keypoints[self.KEYPOINTS['RIGHT_ELBOW'], :2]

        # Vector from shoulder to elbow (upper arm)
        upper_arm = elbow - shoulder

        # Calculate angle relative to vertical (image y-axis)
        if upper_arm[1] == 0:  # Handle horizontal arm case
            return 90.0
        angle = np.degrees(np.arctan2(upper_arm[0], upper_arm[1]))
        return abs(angle)  # Return absolute value for abduction angle

    def process_video_frame(self, frame: np.ndarray, timestamp: float, frame_count: int = 0) -> Tuple[
        np.ndarray, Optional[Dict]]:
        # Process a single video frame for analysis
        if frame is None:
            return np.zeros((300, 400, 3), dtype=np.uint8), None

        try:
            # Run pose detection with GPU acceleration
            with torch.no_grad():  # Disable gradient calculation for inference
                pose_results = self.pose_model(frame, conf=self.conf_threshold)[0]

            keypoints_data = pose_results.keypoints.data.cpu()

            if not self.validate_keypoints(keypoints_data):
                return frame.copy(), None

            # Initialize or update tracked player
            self.update_tracked_player(keypoints_data)

            try:
                return self.analyze_frame(frame, keypoints_data, timestamp)
            except Exception as e:
                print(f"Frame analysis error: {str(e)}")
                return frame.copy(), None

        except Exception as e:
            print(f"Pose detection error: {str(e)}")
            return frame.copy(), None

    def validate_keypoints(self, keypoints_data: torch.Tensor) -> bool:
        # Validate keypoints data structure
        return (
                len(keypoints_data) > 0 and
                keypoints_data.shape[1] > max(self.KEYPOINTS.values())
        )

    def update_tracked_player(self, keypoints_data: torch.Tensor) -> None:
        # Initialize or maintain tracking of target player
        if self.tracked_id is None:
            valid_players = []
            for i, kp in enumerate(keypoints_data):
                if kp.shape[0] > self.KEYPOINTS['RIGHT_WRIST']:
                    try:
                        conf = kp[self.KEYPOINTS['RIGHT_WRIST'], 2].item()
                        valid_players.append((i, conf))
                    except IndexError:
                        continue

            if valid_players:
                self.tracked_id = max(valid_players, key=lambda x: x[1])[0]
        elif self.tracked_id >= len(keypoints_data):
            # Reset tracking if tracked ID is no longer valid
            self.tracked_id = None

    def analyze_frame(self, frame: np.ndarray, keypoints_data: torch.Tensor, timestamp: float) -> Tuple[
        np.ndarray, Optional[Dict]]:
        # Perform frame analysis and visualization
        if self.tracked_id is None or self.tracked_id >= len(keypoints_data):
            self.tracked_id = None
            return frame.copy(), None

        keypoints = keypoints_data[self.tracked_id].numpy()
        if keypoints.shape[0] <= self.KEYPOINTS['RIGHT_WRIST']:
            return frame.copy(), None

        # Detect shuttlecock and calculate metrics
        shuttle_box = self.detect_shuttlecock(frame)
        wrist_point = keypoints[self.KEYPOINTS['RIGHT_WRIST'], :2]
        current_distance = calculate_shuttle_distance(shuttle_box, wrist_point)

        # Calculate angles and shot phase
        grip_angle = self.calculate_grip_angle(keypoints)
        shoulder_angle = self.calculate_shoulder_angle(keypoints)
        shot_phase = self.determine_shot_phase(
            current_distance,
            grip_angle,
            shoulder_angle
        )

        # Update tracking state
        self.update_tracking_state(frame, keypoints, shot_phase, timestamp, current_distance)

        # Generate analysis results
        analysis = FrameAnalysis(
            timestamp=timestamp,
            grip_angle=grip_angle,
            shoulder_angle=shoulder_angle,
            shot_type=self.target_shot_type,
            shot_phase=shot_phase,
            within_threshold=shot_phase is not None,
            keypoints=keypoints
        )

        # Save last analysis for potential frame skipping
        self.last_analysis = analysis

        # Update key frames based on current detections
        self.update_key_frames_from_detections()

        return self.visualize_analysis(frame, analysis, shuttle_box), analysis.__dict__

    def determine_shot_phase(self, current_distance: float, grip_angle: float,
                             shoulder_angle: float) -> Optional[str]:
        # Determine current phase of the shot using cached thresholds
        thresholds = self._thresholds_cache

        if not thresholds:
            print(f"Invalid shot type: {self.target_shot_type}")
            return None

        # Check for hit moment
        if current_distance < self.last_shuttle_distance:
            if is_within_threshold(grip_angle, shoulder_angle, thresholds['hit']):
                return 'hit'

        # Check other phases
        for phase in ['pre_hit', 'follow_through']:
            if is_within_threshold(grip_angle, shoulder_angle, thresholds[phase]):
                return phase

        return None

    def update_tracking_state(self, frame: np.ndarray, keypoints: np.ndarray, phase: Optional[str], timestamp: float,
                              current_distance: float) -> None:
        # Update tracking state variables and store phase detections
        if phase:
            # Add frame to phase detections list
            self.phase_detections[phase].append({
                'frame': frame.copy(),
                'keypoints': keypoints,
                'timestamp': timestamp
            })
        self.last_shuttle_distance = current_distance
        self.prev_keypoints = keypoints
        self.prev_time = timestamp

    def update_key_frames_from_detections(self) -> None:
        # Update key frames based on accumulated phase detections
        # Pre-hit: Get the middle frame between first and last detection
        if len(self.phase_detections['pre_hit']) > 1:
            middle_idx = len(self.phase_detections['pre_hit']) // 2
            self.key_frames['pre_hit'] = self.phase_detections['pre_hit'][middle_idx]
        elif len(self.phase_detections['pre_hit']) == 1:
            self.key_frames['pre_hit'] = self.phase_detections['pre_hit'][0]

        # Hit: Get the last frame
        if len(self.phase_detections['hit']) > 0:
            self.key_frames['hit'] = self.phase_detections['hit'][-1]

        # Follow-through: Get the last frame
        if len(self.phase_detections['follow_through']) > 0:
            self.key_frames['follow_through'] = self.phase_detections['follow_through'][-1]

    def visualize_analysis(self, frame: np.ndarray, analysis: FrameAnalysis,
                           shuttle_box: Optional[np.ndarray]) -> np.ndarray:
        # Generate visualization overlay for analysis results
        viz_frame = frame.copy()
        keypoints = analysis.keypoints

        # Draw skeletal connections using pre-computed connections
        connections = [
            (self.KEYPOINTS['LEFT_SHOULDER'], self.KEYPOINTS['RIGHT_SHOULDER']),
            (self.KEYPOINTS['RIGHT_SHOULDER'], self.KEYPOINTS['RIGHT_ELBOW']),
            (self.KEYPOINTS['RIGHT_ELBOW'], self.KEYPOINTS['RIGHT_WRIST'])
        ]

        # Use pre-computed phase colors
        current_color = self.PHASE_COLORS.get(analysis.shot_phase, self.PHASE_COLORS['default'])

        draw_skeleton(viz_frame, keypoints, connections, current_color)
        draw_shuttlecock(viz_frame, shuttle_box)
        draw_info_overlay(viz_frame, analysis)

        return viz_frame

    def get_technique_summary(self) -> Dict:
        # Generate comprehensive analysis summary
        if not all(self.key_frames.values()):
            return {
                'status': 'incomplete',
                'message': 'Not all key frames captured'
            }

        try:
            return {
                'status': 'complete',
                'shot_type': self.target_shot_type,
                'key_frames': self.key_frames,
                'timing': {
                    'preparation_time':
                        self.key_frames['hit']['timestamp'] - self.key_frames['pre_hit']['timestamp'],
                    'follow_through_time':
                        self.key_frames['follow_through']['timestamp'] - self.key_frames['hit']['timestamp']
                }
            }
        except KeyError as e:
            print(f"Key frame data error: {str(e)}")
            return {
                'status': 'error',
                'message': 'Invalid key frame data'
            }