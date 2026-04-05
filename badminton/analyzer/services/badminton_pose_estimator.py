import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from functools import lru_cache

import cv2
import numpy as np
import torch
from django.conf import settings

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency for local test environments
    YOLO = None  # type: ignore[assignment]

# Basic constants
EPSILON = 1e-8
RADIANS_TO_DEGREES = 180.0 / math.pi


def _safe_point(keypoints: np.ndarray, index: int, confidence_threshold: float = 0.2) -> Optional[np.ndarray]:
    """Return a 2D keypoint when it is available and confident enough."""
    try:
        if keypoints is None or index < 0 or index >= len(keypoints):
            return None

        point = keypoints[index]
        if len(point) < 2:
            return None

        if len(point) >= 3 and point[2] < confidence_threshold:
            return None

        return np.array(point[:2], dtype=np.float32)
    except Exception:
        return None


def _distance_between_points(point_a: Optional[np.ndarray], point_b: Optional[np.ndarray]) -> Optional[float]:
    """Calculate Euclidean distance between two 2D points."""
    if point_a is None or point_b is None:
        return None

    try:
        return float(np.linalg.norm(point_a[:2] - point_b[:2]))
    except Exception:
        return None


def _joint_angle(hip: Optional[np.ndarray], knee: Optional[np.ndarray], ankle: Optional[np.ndarray]) -> Optional[float]:
    """Calculate the angle at the knee joint in degrees."""
    if hip is None or knee is None or ankle is None:
        return None

    try:
        thigh = hip[:2] - knee[:2]
        calf = ankle[:2] - knee[:2]

        thigh_norm = np.linalg.norm(thigh)
        calf_norm = np.linalg.norm(calf)
        if thigh_norm < EPSILON or calf_norm < EPSILON:
            return None

        cosine = np.clip(np.dot(thigh, calf) / (thigh_norm * calf_norm), -1.0, 1.0)
        return float(math.degrees(math.acos(cosine)))
    except Exception:
        return None


@dataclass
class FrameAnalysis:
    """Stores analysis results for a single frame - using __slots__ for memory efficiency"""
    __slots__ = ('timestamp', 'grip_angle', 'shoulder_angle', 'shot_type', 'shot_phase', 'within_threshold', 'keypoints')
    
    timestamp: float
    grip_angle: float
    shoulder_angle: float
    shot_type: str
    shot_phase: Optional[str]
    within_threshold: bool
    keypoints: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert FrameAnalysis to dictionary since __slots__ removes __dict__"""
        try:
            # Safely handle keypoints conversion
            keypoints_list = []
            if hasattr(self, 'keypoints') and self.keypoints is not None:
                if isinstance(self.keypoints, np.ndarray):
                    keypoints_list = self.keypoints.tolist()
                elif isinstance(self.keypoints, list):
                    keypoints_list = self.keypoints
            
            return {
                'timestamp': getattr(self, 'timestamp', 0.0),
                'grip_angle': getattr(self, 'grip_angle', 0.0),
                'shoulder_angle': getattr(self, 'shoulder_angle', 0.0),
                'shot_type': getattr(self, 'shot_type', 'unknown'),
                'shot_phase': getattr(self, 'shot_phase', None),
                'within_threshold': getattr(self, 'within_threshold', False),
                'keypoints': keypoints_list
            }
        except Exception as e:
            print(f"Error in FrameAnalysis.to_dict(): {e}")
            # Return minimal safe dictionary
            return {
                'timestamp': 0.0,
                'grip_angle': 0.0,
                'shoulder_angle': 0.0,
                'shot_type': 'unknown',
                'shot_phase': None,
                'within_threshold': False,
                'keypoints': []
            }
    
    def __getstate__(self):
        """Support for pickling with __slots__"""
        try:
            return {slot: getattr(self, slot, None) for slot in self.__slots__}
        except Exception:
            return {}
    
    def __setstate__(self, state):
        """Support for unpickling with __slots__"""
        try:
            for slot, value in state.items():
                if slot in self.__slots__:
                    setattr(self, slot, value)
        except Exception:
            pass


def calculate_shuttle_distance(shuttle_box: Optional[np.ndarray], wrist_point: np.ndarray) -> float:
    """Calculate distance between shuttlecock and wrist point"""
    if shuttle_box is None or wrist_point is None or len(shuttle_box) < 4:
        return float('inf')
    
    if len(wrist_point) < 2:
        return float('inf')

    try:
        center_x = (shuttle_box[0] + shuttle_box[2]) * 0.5
        center_y = (shuttle_box[1] + shuttle_box[3]) * 0.5
        
        dx = center_x - wrist_point[0]
        dy = center_y - wrist_point[1]
        return math.sqrt(dx * dx + dy * dy)
    except Exception:
        return float('inf')


def calculate_shuttle_distance_squared(shuttle_box: Optional[np.ndarray], wrist_point: np.ndarray) -> float:
    """Calculate squared distance for comparison purposes"""
    if shuttle_box is None or wrist_point is None or len(shuttle_box) < 4:
        return float('inf')
    
    if len(wrist_point) < 2:
        return float('inf')

    try:
        center_x = (shuttle_box[0] + shuttle_box[2]) * 0.5
        center_y = (shuttle_box[1] + shuttle_box[3]) * 0.5
        dx = center_x - wrist_point[0]
        dy = center_y - wrist_point[1]
        return dx * dx + dy * dy
    except Exception:
        return float('inf')


@lru_cache(maxsize=128)
def _cached_threshold_check(grip_angle: float, shoulder_angle: float, 
                           ideal_grip: float, ideal_shoulder: float, tolerance: float) -> bool:
    """Cached threshold checking for repeated calculations"""
    return (abs(grip_angle - ideal_grip) <= tolerance and 
            abs(shoulder_angle - ideal_shoulder) <= tolerance)


def is_within_threshold(grip_angle: float, shoulder_angle: float, threshold: Dict[str, float]) -> bool:
    """Optimized threshold checking with caching and early returns"""
    # Quick validation - early return for invalid inputs
    if not threshold or len(threshold) < 3:
        return False
    
    # Check for required keys with single lookup
    grip_threshold = threshold.get('grip')
    shoulder_threshold = threshold.get('shoulder')
    tolerance = threshold.get('tolerance')
    
    if grip_threshold is None or shoulder_threshold is None or tolerance is None:
        return False

    # Fast numeric validation
    if (not isinstance(grip_angle, (int, float)) or 
        not isinstance(shoulder_angle, (int, float)) or
        not isinstance(tolerance, (int, float))):
        return False
    
    # Use cached function for repeated calculations
    try:
        return _cached_threshold_check(
            float(grip_angle), float(shoulder_angle),
            float(grip_threshold), float(shoulder_threshold), float(tolerance)
        )
    except (ValueError, TypeError):
        return False


def draw_modern_skeleton(frame: np.ndarray, keypoints: np.ndarray, connections: List[Tuple[int, int]],
                        color: Tuple[int, int, int], phase: str = 'default') -> None:
    """Modern skeleton drawing with clean, visible design"""
    conf_threshold = 0.5
    
    # Calculate balanced sizing - visible but not overwhelming
    h, w = frame.shape[:2]
    scale_factor = min(w, h) / 720.0  # Base on 720p for good visibility
    scale_factor = max(0.6, min(1.4, scale_factor))  # Reasonable scaling range
    
    # Modern color palette with vibrant, distinctive colors
    phase_styles = {
        'pre_hit': {
            'color': (100, 255, 100),  # Bright green
            'thickness': max(2, int(2 * scale_factor)),
            'joint_size': max(3, int(4 * scale_factor))
        },
        'hit': {
            'color': (100, 100, 255),  # Bright blue
            'thickness': max(2, int(2 * scale_factor)),
            'joint_size': max(3, int(4 * scale_factor))
        },
        'follow_through': {
            'color': (255, 100, 100),  # Bright red
            'thickness': max(2, int(2 * scale_factor)),
            'joint_size': max(3, int(4 * scale_factor))
        },
        'default': {
            'color': (200, 200, 200),  # Light gray
            'thickness': max(2, int(2 * scale_factor)),
            'joint_size': max(3, int(4 * scale_factor))
        }
    }
    
    style = phase_styles.get(phase, phase_styles['default'])
    
    # Draw clean, visible skeleton
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
            
        if keypoints[start_idx, 2] > conf_threshold and keypoints[end_idx, 2] > conf_threshold:
            try:
                start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
                end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
                
                # Anti-aliased line with good visibility
                cv2.line(frame, start_point, end_point, style['color'], style['thickness'], cv2.LINE_AA)
                
                # Anti-aliased joint dots
                joint_size = style['joint_size']
                cv2.circle(frame, start_point, joint_size, style['color'], -1, cv2.LINE_AA)
                cv2.circle(frame, end_point, joint_size, style['color'], -1, cv2.LINE_AA)
                
            except (IndexError, ValueError):
                continue


def draw_modern_shuttlecock(frame: np.ndarray, shuttle_box: Optional[np.ndarray]) -> None:
    """Modern shuttlecock drawing with clean, visible design"""
    if shuttle_box is not None and len(shuttle_box) >= 4:
        try:
            x1, y1, x2, y2 = map(int, shuttle_box)
            
            # Calculate balanced sizing
            h, w = frame.shape[:2]
            scale_factor = min(w, h) / 720.0
            scale_factor = max(0.6, min(1.4, scale_factor))
            
            # Clean line thickness
            line_thickness = max(2, int(2 * scale_factor))
            
            # Vibrant bounding box with anti-aliasing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 200, 255), line_thickness, cv2.LINE_AA)
            
            # Visible crosshair with anti-aliasing
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cross_size = max(8, int(12 * scale_factor))
            
            # Anti-aliased crosshair
            cv2.line(frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (100, 200, 255), line_thickness, cv2.LINE_AA)
            cv2.line(frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (100, 200, 255), line_thickness, cv2.LINE_AA)
            
            # Readable label
            label_text = "SHUTTLE"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, 0.5 * scale_factor)
            text_thickness = max(1, int(1 * scale_factor))
            
            # Position label above the box
            label_x = max(0, x1)
            label_y = max(15, y1 - 8)
            
            # Ensure label stays within frame bounds
            label_x = min(label_x, w - 60)
            
            # Anti-aliased text
            cv2.putText(frame, label_text, (label_x, label_y), font, font_scale, (100, 200, 255), text_thickness, cv2.LINE_AA)
            
        except (ValueError, IndexError):
            pass


def draw_modern_info_overlay(frame: np.ndarray, analysis: FrameAnalysis,
                             footwork_metrics: Optional[Dict[str, Any]] = None,
                             guidance: Optional[Dict[str, Any]] = None) -> None:
    """Modern info overlay with clean, readable design"""
    h, w = frame.shape[:2]
    
    # Balanced sizing for readability
    scale_factor = min(w, h) / 720.0
    scale_factor = max(0.6, min(1.4, scale_factor))
    
    # Reasonably sized panel
    panel_width = int(180 * scale_factor)
    panel_height = int(90 * scale_factor)
    panel_x = int(15 * scale_factor)
    panel_y = int(15 * scale_factor)
    
    # Clean background with good visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Clean border
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                 (120, 120, 120), 1)
    
    # Vibrant status colors
    phase_colors = {
        'pre_hit': (100, 255, 100),      # Bright green
        'hit': (100, 100, 255),          # Bright blue  
        'follow_through': (255, 100, 100),  # Bright red
        'default': (200, 200, 200)       # Light gray
    }
    
    phase = analysis.shot_phase or 'default'
    phase_color = phase_colors.get(phase, phase_colors['default'])
    
    # Visible status indicator
    dot_x = panel_x + int(12 * scale_factor)
    dot_y = panel_y + int(18 * scale_factor)
    dot_size = max(4, int(5 * scale_factor))
    
    cv2.circle(frame, (dot_x, dot_y), dot_size, phase_color, -1, cv2.LINE_AA)
    
    # Anti-aliased text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, 0.5 * scale_factor)
    thickness = max(1, int(1 * scale_factor))
    
    # Phase text
    text_x = dot_x + int(15 * scale_factor)
    phase_display = phase.replace('_', ' ').upper()
    if len(phase_display) > 8:
        phase_display = phase_display[:8]
    cv2.putText(frame, phase_display, (text_x, panel_y + int(22 * scale_factor)), 
               font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
    
    # Angle information
    grip_text = f"GRIP: {analysis.grip_angle:.0f}deg"
    shoulder_text = f"ARM: {analysis.shoulder_angle:.0f}deg"
    
    cv2.putText(frame, grip_text, (text_x, panel_y + int(45 * scale_factor)), 
               font, font_scale * 0.8, (200, 200, 200), thickness, cv2.LINE_AA)
    cv2.putText(frame, shoulder_text, (text_x, panel_y + int(65 * scale_factor)), 
               font, font_scale * 0.8, (200, 200, 200), thickness, cv2.LINE_AA)
    
    # Status indicator
    status_color = (100, 255, 100) if analysis.within_threshold else (255, 100, 100)
    status_text = "OK" if analysis.within_threshold else "X"
    status_x = panel_x + panel_width - int(25 * scale_factor)
    cv2.putText(frame, status_text, (status_x, panel_y + int(22 * scale_factor)), 
               font, font_scale, status_color, thickness, cv2.LINE_AA)

    if not footwork_metrics:
        return

    guidance = guidance or {}
    phase_display = phase.replace('_', ' ').upper()
    footwork_panel_width = min(int(320 * scale_factor), max(1, w - (panel_x * 2)))
    footwork_panel_height = min(int(132 * scale_factor), max(1, h - (panel_y * 2)))
    footwork_panel_x = w - footwork_panel_width - panel_x
    footwork_panel_y = panel_y

    footwork_overlay = frame.copy()
    cv2.rectangle(
        footwork_overlay,
        (footwork_panel_x, footwork_panel_y),
        (footwork_panel_x + footwork_panel_width, footwork_panel_y + footwork_panel_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(footwork_overlay, 0.62, frame, 0.38, 0, frame)

    cv2.rectangle(
        frame,
        (footwork_panel_x, footwork_panel_y),
        (footwork_panel_x + footwork_panel_width, footwork_panel_y + footwork_panel_height),
        (120, 120, 120),
        1
    )

    marker_x = footwork_panel_x + int(12 * scale_factor)
    marker_y = footwork_panel_y + int(18 * scale_factor)
    cv2.circle(frame, (marker_x, marker_y), max(4, int(5 * scale_factor)), phase_color, -1, cv2.LINE_AA)

    footwork_font_scale = max(0.35, 0.42 * scale_factor)
    footwork_thickness = max(1, int(1 * scale_factor))
    footwork_text_x = marker_x + int(15 * scale_factor)

    def shorten_text(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[:limit - 3].rstrip() + '...'

    pattern_text = shorten_text(guidance.get('pattern', 'split step -> load -> recover').replace(' -> ', ' | '), 58)
    footwork_flags = footwork_metrics.get('flags') or []
    cue_text = footwork_flags[0] if footwork_flags else footwork_metrics.get('phase_note', 'Keep the feet active and recover quickly.')
    cue_text = shorten_text(str(cue_text), 72)

    stance_ratio = footwork_metrics.get('stance_width_ratio')
    knee_angle = footwork_metrics.get('average_knee_angle')
    ready_target = footwork_metrics.get('ideal', {}).get('ready_stance_ratio', guidance.get('ideal_ready_stance_ratio', 1.05))
    recovery_target = footwork_metrics.get('ideal', {}).get('recovery_ratio_target', guidance.get('recovery_ratio_target', 0.75))

    header_text = f"{phase_display} FOOTWORK"
    cv2.putText(
        frame,
        header_text,
        (footwork_text_x, footwork_panel_y + int(22 * scale_factor)),
        font,
        footwork_font_scale,
        (240, 240, 240),
        footwork_thickness,
        cv2.LINE_AA
    )

    lines = [
        f"PATTERN: {pattern_text}",
        f"CUE: {cue_text}",
        f"STANCE: {stance_ratio:.2f}x" if isinstance(stance_ratio, (int, float)) else "STANCE: --",
        f"KNEE: {knee_angle:.0f}deg" if isinstance(knee_angle, (int, float)) else "KNEE: --",
        f"TARGET: {ready_target:.2f}x | RECOVER: {recovery_target:.2f}x"
    ]

    for line_index, line_text in enumerate(lines):
        line_y = footwork_panel_y + int((42 + (line_index * 18)) * scale_factor)
        cv2.putText(
            frame,
            shorten_text(line_text, 64),
            (footwork_text_x, line_y),
            font,
            footwork_font_scale * 0.86,
            (215, 215, 215),
            footwork_thickness,
            cv2.LINE_AA
        )


def draw_modern_footwork_tracking(frame: np.ndarray, keypoints: np.ndarray,
                                  footwork_metrics: Optional[Dict[str, Any]] = None,
                                  guidance: Optional[Dict[str, Any]] = None,
                                  phase: str = 'default') -> None:
    """Highlight lower-body stance width and recovery alignment on the frame."""
    if keypoints is None or len(keypoints) <= 16:
        return

    left_ankle = _safe_point(keypoints, 15)
    right_ankle = _safe_point(keypoints, 16)
    left_hip = _safe_point(keypoints, 11)
    right_hip = _safe_point(keypoints, 12)

    if left_ankle is None or right_ankle is None:
        return

    h, w = frame.shape[:2]
    scale_factor = min(w, h) / 720.0
    scale_factor = max(0.6, min(1.4, scale_factor))

    phase_colors = {
        'pre_hit': (100, 255, 100),
        'hit': (100, 100, 255),
        'follow_through': (255, 100, 100),
        'default': (200, 200, 200)
    }
    phase_color = phase_colors.get(phase, phase_colors['default'])

    stance_ratio = footwork_metrics.get('stance_width_ratio') if footwork_metrics else None
    ready_low, ready_high = guidance.get('ready_stance_range', (0.9, 1.25)) if guidance else (0.9, 1.25)

    if isinstance(stance_ratio, (int, float)):
        if stance_ratio < ready_low:
            tracking_color = (80, 80, 255)
        elif stance_ratio > ready_high:
            tracking_color = (0, 215, 255)
        else:
            tracking_color = (100, 255, 100)
    else:
        tracking_color = phase_color

    left_point = (int(left_ankle[0]), int(left_ankle[1]))
    right_point = (int(right_ankle[0]), int(right_ankle[1]))

    line_thickness = max(2, int(3 * scale_factor))
    marker_radius = max(5, int(6 * scale_factor))
    cv2.line(frame, left_point, right_point, tracking_color, line_thickness, cv2.LINE_AA)
    cv2.circle(frame, left_point, marker_radius, tracking_color, -1, cv2.LINE_AA)
    cv2.circle(frame, right_point, marker_radius, tracking_color, -1, cv2.LINE_AA)
    cv2.circle(frame, left_point, marker_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, right_point, marker_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    if left_hip is not None and right_hip is not None:
        hip_center = ((left_hip + right_hip) * 0.5).astype(np.int32)
        ankle_center = ((left_ankle + right_ankle) * 0.5).astype(np.int32)
        cv2.line(
            frame,
            (int(hip_center[0]), int(hip_center[1])),
            (int(ankle_center[0]), int(ankle_center[1])),
            (180, 180, 180),
            max(1, int(2 * scale_factor)),
            cv2.LINE_AA
        )

    midpoint_x = int((left_point[0] + right_point[0]) * 0.5)
    midpoint_y = int((left_point[1] + right_point[1]) * 0.5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.45 * scale_factor)
    thickness = max(1, int(1 * scale_factor))
    label_text = 'BASE'
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    label_x = max(0, min(w - text_width - 1, midpoint_x - (text_width // 2)))
    label_y = max(text_height + 2, midpoint_y - int(10 * scale_factor))
    cv2.rectangle(
        frame,
        (label_x - 2, label_y - text_height - 2),
        (label_x + text_width + 2, label_y + 2),
        (0, 0, 0),
        -1
    )
    cv2.putText(frame, label_text, (label_x, label_y), font, font_scale, tracking_color, thickness, cv2.LINE_AA)


def draw_modern_connection_line(frame: np.ndarray, start_point: Tuple[int, int], 
                               end_point: Tuple[int, int], distance: float) -> None:
    """Modern connection line with clean, visible design"""
    # Balanced sizing
    h, w = frame.shape[:2]
    scale_factor = min(w, h) / 720.0
    scale_factor = max(0.6, min(1.4, scale_factor))
    
    # Vibrant color gradient based on distance
    max_distance = 200
    normalized_distance = min(distance / max_distance, 1.0)
    
    # More vibrant color gradient
    red = int(100 + 155 * normalized_distance)
    green = int(100 + 155 * (1 - normalized_distance))
    blue = 100
    
    line_color = (blue, green, red)
    
    # Visible line thickness
    thickness = max(1, int(2 * scale_factor))
    
    # Calculate clean dashed line
    line_length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    dash_length = max(10, int(12 * scale_factor))
    gap_length = max(5, int(6 * scale_factor))
    total_dash_unit = dash_length + gap_length
    num_dashes = int(line_length / total_dash_unit)
    
    # Draw clean dashed line
    for i in range(num_dashes):
        t1 = (i * total_dash_unit) / line_length
        t2 = min(((i * total_dash_unit) + dash_length) / line_length, 1.0)
        
        if t2 > t1:
            x1 = int(start_point[0] + t1 * (end_point[0] - start_point[0]))
            y1 = int(start_point[1] + t1 * (end_point[1] - start_point[1]))
            x2 = int(start_point[0] + t2 * (end_point[0] - start_point[0]))
            y2 = int(start_point[1] + t2 * (end_point[1] - start_point[1]))
            
            cv2.line(frame, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)
    
    # Readable distance display
    mid_x = (start_point[0] + end_point[0]) // 2
    mid_y = (start_point[1] + end_point[1]) // 2
    
    # Clean distance text
    distance_text = f"{distance:.0f}px"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, 0.4 * scale_factor)
    
    # Position text with good visibility
    text_x = mid_x + int(8 * scale_factor)
    text_y = mid_y - int(5 * scale_factor)
    
    # Clean text with subtle background
    (text_width, text_height), baseline = cv2.getTextSize(distance_text, font, font_scale, 1)
    cv2.rectangle(frame, (text_x - 2, text_y - text_height - 2), 
                 (text_x + text_width + 2, text_y + 2), (0, 0, 0), -1)
    cv2.putText(frame, distance_text, (text_x, text_y), font, font_scale, (240, 240, 240), 1, cv2.LINE_AA)


# Keep backward compatibility with simple wrappers
def draw_skeleton_optimized(frame: np.ndarray, keypoints: np.ndarray, connections: List[Tuple[int, int]],
                           color: Tuple[int, int, int]) -> None:
    """Legacy wrapper for backward compatibility"""
    draw_modern_skeleton(frame, keypoints, connections, color)


def draw_shuttlecock_optimized(frame: np.ndarray, shuttle_box: Optional[np.ndarray]) -> None:
    """Legacy wrapper for backward compatibility"""
    draw_modern_shuttlecock(frame, shuttle_box)


def draw_info_overlay_optimized(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    """Legacy wrapper for backward compatibility"""
    draw_modern_info_overlay(frame, analysis)


# Keep original functions for backward compatibility
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, connections: List[Tuple[int, int]],
                  color: Tuple[int, int, int]) -> None:
    draw_skeleton_optimized(frame, keypoints, connections, color)


def draw_shuttlecock(frame: np.ndarray, shuttle_box: Optional[np.ndarray]) -> None:
    draw_shuttlecock_optimized(frame, shuttle_box)


def draw_info_overlay(frame: np.ndarray, analysis: FrameAnalysis) -> None:
    draw_info_overlay_optimized(frame, analysis)


class BadmintonVideoTracker:
    """Badminton video tracker for pose analysis"""
    
    # Keypoint indices
    KEYPOINTS = {
        'LEFT_SHOULDER': 5,
        'RIGHT_SHOULDER': 6,
        'LEFT_ELBOW': 7,
        'RIGHT_ELBOW': 8,
        'LEFT_WRIST': 9,
        'RIGHT_WRIST': 10,
        'LEFT_HIP': 11,
        'RIGHT_HIP': 12,
        'LEFT_KNEE': 13,
        'RIGHT_KNEE': 14,
        'LEFT_ANKLE': 15,
        'RIGHT_ANKLE': 16
    }

    # Default connections for skeleton visualization
    DEFAULT_CONNECTIONS = [
        (5, 6),  # Left to right shoulder
        (6, 8),  # Right shoulder to right elbow
        (8, 10)  # Right elbow to right wrist
    ]

    # Phase colors for visualization
    PHASE_COLORS = {
        'pre_hit': (0, 255, 0),  # Green
        'hit': (0, 0, 255),  # Red
        'follow_through': (255, 0, 0),  # Blue
        'default': (128, 128, 128)  # Gray
    }

    # Shot thresholds
    SHOT_THRESHOLDS = {
        'smash': {
            'pre_hit': {'grip': 90.0, 'shoulder': 45.0, 'tolerance': 12.0},
            'hit': {'grip': 165.0, 'shoulder': 105.0, 'tolerance': 12.0},
            'follow_through': {'grip': 150.0, 'shoulder': 70.0, 'tolerance': 12.0}
        },
        'clear': {
            'pre_hit': {'grip': 150.0, 'shoulder': 60.0, 'tolerance': 12.0},
            'hit': {'grip': 170.0, 'shoulder': 70.0, 'tolerance': 8.0},
            'follow_through': {'grip': 140.0, 'shoulder': 50.0, 'tolerance': 12.0}
        },
        'drop_shot': {
            'pre_hit': {'grip': 95.0, 'shoulder': 35.0, 'tolerance': 12.0},
            'hit': {'grip': 105.0, 'shoulder': 45.0, 'tolerance': 8.0},
            'follow_through': {'grip': 85.0, 'shoulder': 25.0, 'tolerance': 12.0}
        }
    }

    FOOTWORK_GUIDANCE = {
        'smash': {
            'pattern': 'split step -> rear-court load -> scissor kick or jump out -> recover',
            'ready_stance_range': (0.95, 1.35),
            'ideal_ready_stance_ratio': 1.10,
            'ideal_knee_angle': 145.0,
            'recovery_ratio_target': 0.65,
            'pre_hit_note': 'Use a quick split step and get side-on before loading the rear leg.',
            'hit_note': 'Explode through the contact with a scissor kick or jump out and keep the landing balanced.',
            'follow_through_note': 'Land wide and recover immediately back to base after contact.'
        },
        'clear': {
            'pattern': 'split step -> chasse / pivot -> rear-court strike -> recover',
            'ready_stance_range': (1.00, 1.40),
            'ideal_ready_stance_ratio': 1.15,
            'ideal_knee_angle': 150.0,
            'recovery_ratio_target': 0.75,
            'pre_hit_note': 'Split step early, then chasse and pivot so you get behind the shuttle.',
            'hit_note': 'Strike from a stable rear-court base with the body turned side-on.',
            'follow_through_note': 'Push back out of the shot and recover into court balance quickly.'
        },
        'drop_shot': {
            'pattern': 'split step -> quick first step -> controlled lunge -> recover',
            'ready_stance_range': (0.80, 1.15),
            'ideal_ready_stance_ratio': 0.95,
            'ideal_knee_angle': 135.0,
            'recovery_ratio_target': 0.85,
            'pre_hit_note': 'Use a small split step and move in with a short, sharp first step.',
            'hit_note': 'Stay low and control the lunge so you can take the shuttle early at the net.',
            'follow_through_note': 'Push off the front leg and get back to the base without pausing.'
        },
        'default': {
            'pattern': 'split step -> load -> strike -> recover',
            'ready_stance_range': (0.90, 1.25),
            'ideal_ready_stance_ratio': 1.05,
            'ideal_knee_angle': 145.0,
            'recovery_ratio_target': 0.75,
            'pre_hit_note': 'Start with a balanced split step and stay light on your feet.',
            'hit_note': 'Keep the legs loaded at contact so you can move out of the shot.',
            'follow_through_note': 'Recover to a neutral base as soon as the shuttle leaves the strings.'
        }
    }

    # Shuttle detection search tuning
    SHUTTLE_SEARCH_PADDING = 24
    SHUTTLE_SEARCH_EXPANSION = 2.0

    def __init__(self, target_shot_type: str = 'smash', model_size: str = 'n'):
        if YOLO is None:
            raise ImportError("ultralytics is required for badminton pose estimation")

        # Initialize models
        models_dir = getattr(settings, 'YOLO_MODELS_DIR', os.path.join(settings.MEDIA_ROOT, 'models'))
        os.makedirs(models_dir, exist_ok=True)

        pose_model_path = os.path.join(models_dir, f'yolo11{model_size}-pose.pt')
        shuttle_model_path = os.path.join(models_dir, 'shuttlecock.pt')

        # Check if models exist
        if not os.path.exists(pose_model_path):
            pose_model_path = f'yolo11{model_size}-pose.pt'
        if not os.path.exists(shuttle_model_path):
            shuttle_model_path = 'shuttlecock.pt'

        # Initialize models
        self.pose_model = YOLO(pose_model_path)
        self.shuttle_model = YOLO(shuttle_model_path)

        # Detection parameters
        self.conf_threshold = 0.5  # For pose detection
        self.iou_threshold = 0.6
        
        # Separate, more restrictive threshold for shuttlecock detection
        self.shuttle_conf_threshold = 0.65  # Higher threshold to reduce false positives

        # Initialize tracking state
        self.target_shot_type = target_shot_type
        self.tracked_id = None
        self.key_frames: Dict[str, Optional[Dict]] = {
            'pre_hit': None,
            'hit': None,
            'follow_through': None
        }

        # Phase detection storage
        self.phase_detections = {
            'pre_hit': [],
            'hit': [],
            'follow_through': []
        }

        # Tracking state variables
        self.last_shuttle_distance = float('inf')
        self.prev_keypoints = None
        self.prev_shuttle_box = None
        self.prev_time = None
        self.last_analysis = None

        # Hit detection tracking
        self.min_shuttle_distance = float('inf')
        self.min_distance_frame = None

        # Sequence state tracking
        self.sequence_state = 'initial'
        self.found_hit = False
        self.awaiting_follow_through = False
        self.prev_person_box = None

        # Keypoint smoothing
        self.keypoint_buffer = []
        self.buffer_size = 3
        
        # Simple caches for performance
        self._roi_cache = {}
        self._thresholds_cache = self.SHOT_THRESHOLDS.get(target_shot_type, {})

    def validate_frame_data(self, frame: np.ndarray, keypoints_data: torch.Tensor) -> bool:
        """Validate input data before processing"""
        try:
            if frame is None or keypoints_data is None:
                return False
            
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                return False
                
            if not isinstance(keypoints_data, torch.Tensor) or keypoints_data.numel() == 0:
                return False
                
            return True
        except Exception:
            return False

    def detect_shuttlecock(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect shuttlecock in frame, focusing on upper half with high confidence"""
        h, w = frame.shape[:2]

        search_frame, search_offset = self._build_shuttle_search_region(frame)
        
        # Run detection with higher confidence threshold
        with torch.no_grad():
            results = self.shuttle_model(search_frame, conf=self.shuttle_conf_threshold, 
                                       iou=self.iou_threshold, classes=[0])

        # Process results with additional filtering
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            if boxes.size == 0 or confidences.size == 0:
                self.prev_shuttle_box = None
                return None
            
            # Additional filtering: only accept very high confidence detections
            high_conf_mask = confidences >= self.shuttle_conf_threshold
            if not np.any(high_conf_mask):
                self.prev_shuttle_box = None
                return None

            if search_offset != (0, 0):
                boxes = self._shift_boxes_to_frame(boxes, search_offset[0], search_offset[1])
            
            # Get best detection among high confidence ones
            filtered_boxes = boxes[high_conf_mask]
            filtered_confidences = confidences[high_conf_mask]
            best_idx = np.argmax(filtered_confidences)
            best_box = self._clip_box_to_bounds(filtered_boxes[best_idx], w, h // 2)

            if best_box is None:
                self.prev_shuttle_box = None
                return None
            
            # Additional size filtering: shuttlecock should be reasonably sized
            box_width = best_box[2] - best_box[0]
            box_height = best_box[3] - best_box[1]
            
            # Reject boxes that are too small (< 10px) or too large (> 200px in any dimension)
            if box_width < 10 or box_height < 10 or box_width > 200 or box_height > 200:
                self.prev_shuttle_box = None
                return None
            
            # Reject boxes with extreme aspect ratios (shuttlecock should be roughly square-ish)
            aspect_ratio = box_width / box_height
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                self.prev_shuttle_box = None
                return None

            self.prev_shuttle_box = best_box
            return best_box

        self.prev_shuttle_box = None
        return None

    def _build_shuttle_search_region(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Build a reduced search region around the previous shuttle detection when possible."""
        h, w = frame.shape[:2]
        top_half_limit = max(1, h // 2)

        if self.prev_shuttle_box is None:
            return frame[:top_half_limit, :], (0, 0)

        try:
            x1, y1, x2, y2 = [float(value) for value in self.prev_shuttle_box[:4]]
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)

            pad_x = max(self.SHUTTLE_SEARCH_PADDING, int(box_width * self.SHUTTLE_SEARCH_EXPANSION))
            pad_y = max(self.SHUTTLE_SEARCH_PADDING, int(box_height * self.SHUTTLE_SEARCH_EXPANSION))

            roi_x1 = max(0, int(x1 - pad_x))
            roi_y1 = max(0, int(y1 - pad_y))
            roi_x2 = min(w, int(x2 + pad_x))
            roi_y2 = min(top_half_limit, int(y2 + pad_y))

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
                top_half_area = w * top_half_limit

                # Only use the ROI if it is meaningfully smaller than the full search area.
                if roi_area < top_half_area * 0.9:
                    return frame[roi_y1:roi_y2, roi_x1:roi_x2], (roi_x1, roi_y1)
        except Exception:
            pass

        return frame[:top_half_limit, :], (0, 0)

    @staticmethod
    def _shift_boxes_to_frame(boxes: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
        """Shift model output boxes back into full-frame coordinates."""
        shifted = boxes.copy()
        shifted[:, [0, 2]] += offset_x
        shifted[:, [1, 3]] += offset_y
        return shifted

    @staticmethod
    def _clip_box_to_bounds(box: np.ndarray, max_width: int, max_height: int) -> Optional[np.ndarray]:
        """Clip a box to frame bounds and reject degenerate results."""
        clipped = np.array(box, dtype=np.float32).copy()
        clipped[0] = max(0.0, min(clipped[0], max_width - 1))
        clipped[1] = max(0.0, min(clipped[1], max_height - 1))
        clipped[2] = max(0.0, min(clipped[2], max_width - 1))
        clipped[3] = max(0.0, min(clipped[3], max_height - 1))

        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None

        return clipped

    def calculate_grip_angle(self, keypoints: np.ndarray) -> float:
        """Calculate grip angle between upper arm and forearm"""
        try:
            wrist_idx = self.KEYPOINTS['RIGHT_WRIST']
            elbow_idx = self.KEYPOINTS['RIGHT_ELBOW']
            shoulder_idx = self.KEYPOINTS['RIGHT_SHOULDER']
            
            wrist = keypoints[wrist_idx, :2]
            elbow = keypoints[elbow_idx, :2]
            shoulder = keypoints[shoulder_idx, :2]

            upper_arm = shoulder - elbow
            forearm = wrist - elbow

            norm_upper = np.linalg.norm(upper_arm)
            norm_forearm = np.linalg.norm(forearm)
            
            if norm_upper < EPSILON or norm_forearm < EPSILON:
                return 0.0

            upper_arm /= norm_upper
            forearm /= norm_forearm

            dot_product = np.dot(upper_arm, forearm)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            angle_rad = math.acos(dot_product)
            return angle_rad * RADIANS_TO_DEGREES
            
        except (IndexError, ValueError):
            return 0.0

    def calculate_shoulder_angle(self, keypoints: np.ndarray) -> float:
        """Calculate shoulder angle"""
        try:
            shoulder = keypoints[self.KEYPOINTS['RIGHT_SHOULDER'], :2]
            elbow = keypoints[self.KEYPOINTS['RIGHT_ELBOW'], :2]

            upper_arm = elbow - shoulder
            norm_upper_arm = np.linalg.norm(upper_arm)
            
            if norm_upper_arm < EPSILON:
                return 0.0

            upper_arm /= norm_upper_arm
            
            # Calculate angle with vertical (0, 1)
            angle_rad = math.acos(np.clip(upper_arm[1], -1.0, 1.0))
            return angle_rad * RADIANS_TO_DEGREES
            
        except (IndexError, ValueError):
            return 0.0

    def _calculate_footwork_phase_metrics(self, keypoints: np.ndarray, phase: str) -> Dict[str, Any]:
        """Extract footwork metrics for a single key frame."""
        guidance = self.FOOTWORK_GUIDANCE.get(self.target_shot_type, self.FOOTWORK_GUIDANCE['default'])

        left_shoulder = _safe_point(keypoints, self.KEYPOINTS['LEFT_SHOULDER'])
        right_shoulder = _safe_point(keypoints, self.KEYPOINTS['RIGHT_SHOULDER'])
        left_hip = _safe_point(keypoints, self.KEYPOINTS['LEFT_HIP'])
        right_hip = _safe_point(keypoints, self.KEYPOINTS['RIGHT_HIP'])
        left_knee = _safe_point(keypoints, self.KEYPOINTS['LEFT_KNEE'])
        right_knee = _safe_point(keypoints, self.KEYPOINTS['RIGHT_KNEE'])
        left_ankle = _safe_point(keypoints, self.KEYPOINTS['LEFT_ANKLE'])
        right_ankle = _safe_point(keypoints, self.KEYPOINTS['RIGHT_ANKLE'])

        shoulder_width = _distance_between_points(left_shoulder, right_shoulder)
        hip_width = _distance_between_points(left_hip, right_hip)
        stance_width = _distance_between_points(left_ankle, right_ankle)

        ankle_span_x = None
        ankle_span_y = None
        if left_ankle is not None and right_ankle is not None:
            ankle_span_x = float(abs(left_ankle[0] - right_ankle[0]))
            ankle_span_y = float(abs(left_ankle[1] - right_ankle[1]))

        stance_width_ratio = None
        if stance_width is not None and shoulder_width is not None and shoulder_width > EPSILON:
            stance_width_ratio = stance_width / shoulder_width

        left_knee_angle = _joint_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = _joint_angle(right_hip, right_knee, right_ankle)
        knee_angles = [angle for angle in [left_knee_angle, right_knee_angle] if isinstance(angle, (int, float))]
        average_knee_angle = float(sum(knee_angles) / len(knee_angles)) if knee_angles else None

        hip_center = None
        if left_hip is not None and right_hip is not None:
            hip_center = ((left_hip + right_hip) * 0.5).tolist()

        ankle_center = None
        if left_ankle is not None and right_ankle is not None:
            ankle_center = ((left_ankle + right_ankle) * 0.5).tolist()

        phase_note_key = f'{phase}_note'
        phase_note = guidance.get(phase_note_key, guidance['pre_hit_note'])

        target_range_low, target_range_high = guidance['ready_stance_range']
        footwork_flags = []
        if phase == 'pre_hit' and stance_width_ratio is not None:
            if stance_width_ratio < target_range_low:
                footwork_flags.append('Base is too narrow for an explosive split step.')
            elif stance_width_ratio > target_range_high:
                footwork_flags.append('Base is too wide, which will slow the first push-off.')

        if phase == 'hit' and average_knee_angle is not None:
            if average_knee_angle > guidance['ideal_knee_angle'] + 12:
                footwork_flags.append('You are too upright at contact; sink a little lower into the lunge or scissor.')
            elif average_knee_angle < guidance['ideal_knee_angle'] - 18:
                footwork_flags.append('You are sitting very deep; stay low but keep the movement compact.')

        if phase == 'follow_through' and stance_width_ratio is not None:
            if stance_width_ratio < target_range_low:
                footwork_flags.append('Recovery foot placement is too tight; re-open the base sooner.')
            elif stance_width_ratio > target_range_high + 0.15:
                footwork_flags.append('Recovery is still too spread out; tighten the landing and push back to base faster.')

        return {
            'phase': phase,
            'stance_width': stance_width,
            'stance_width_ratio': stance_width_ratio,
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'ankle_span_x': ankle_span_x,
            'ankle_span_y': ankle_span_y,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
            'average_knee_angle': average_knee_angle,
            'hip_center': hip_center,
            'ankle_center': ankle_center,
            'phase_note': phase_note,
            'flags': footwork_flags,
            'ideal': {
                'ready_stance_ratio': guidance['ideal_ready_stance_ratio'],
                'knee_angle': guidance['ideal_knee_angle'],
                'recovery_ratio_target': guidance['recovery_ratio_target']
            }
        }

    def _build_footwork_summary(self, phase_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create a shot-specific footwork summary from the key frame metrics."""
        guidance = self.FOOTWORK_GUIDANCE.get(self.target_shot_type, self.FOOTWORK_GUIDANCE['default'])
        pre_hit = phase_metrics.get('pre_hit', {})
        hit = phase_metrics.get('hit', {})
        follow_through = phase_metrics.get('follow_through', {})

        ready_low, ready_high = guidance['ready_stance_range']
        focus_areas = []
        recommendations = []
        phase_feedback = {
            'pre_hit': pre_hit.get('phase_note', guidance['pre_hit_note']),
            'hit': hit.get('phase_note', guidance['hit_note']),
            'follow_through': follow_through.get('phase_note', guidance['follow_through_note'])
        }

        for phase_name, metrics in [('pre_hit', pre_hit), ('hit', hit), ('follow_through', follow_through)]:
            stance_ratio = metrics.get('stance_width_ratio')
            knee_angle = metrics.get('average_knee_angle')

            if phase_name == 'pre_hit' and stance_ratio is not None:
                if stance_ratio < ready_low:
                    focus_areas.append('split step balance')
                    recommendations.append('Widen the split step and land lighter on the balls of your feet.')
                elif stance_ratio > ready_high:
                    focus_areas.append('first-step explosiveness')
                    recommendations.append('Tighten the ready stance so you can push off faster.')

            if phase_name == 'hit' and knee_angle is not None:
                if knee_angle > guidance['ideal_knee_angle'] + 10:
                    focus_areas.append('loading into the shot')
                    recommendations.append('Sink lower into the lunge or scissor so the legs can drive the shot.')
                elif knee_angle < guidance['ideal_knee_angle'] - 15:
                    focus_areas.append('compact contact posture')
                    recommendations.append('Stay low, but keep the contact compact and balanced.')

            if phase_name == 'follow_through' and stance_ratio is not None:
                if stance_ratio < ready_low:
                    focus_areas.append('recovery positioning')
                    recommendations.append('Push off the racket leg sooner and reopen your base after contact.')
                elif stance_ratio > ready_high + 0.15:
                    focus_areas.append('recovery speed')
                    recommendations.append('Recover back to a neutral base faster after the follow-through.')

        if not focus_areas:
            focus_areas.extend([item for item in [
                'split step timing',
                'landing balance',
                'recovery speed'
            ] if item not in focus_areas])

        if not recommendations:
            recommendations = [
                'Keep the feet active between strokes and stay ready on the balls of your feet.',
                'Recover to a neutral base immediately after each shot.'
            ]

        overall_parts = [f"Your {self.target_shot_type.replace('_', ' ')} footwork follows the expected {guidance['pattern']} rhythm."]
        if phase_feedback['pre_hit']:
            overall_parts.append(phase_feedback['pre_hit'])
        if phase_feedback['hit']:
            overall_parts.append(phase_feedback['hit'])
        if phase_feedback['follow_through']:
            overall_parts.append(phase_feedback['follow_through'])

        return {
            'overall_assessment': ' '.join(overall_parts).strip(),
            'movement_pattern': guidance['pattern'],
            'phase_metrics': phase_metrics,
            'phase_feedback': phase_feedback,
            'focus_areas': focus_areas[:3],
            'recommendations': recommendations[:3],
            'targets': {
                'ready_stance_ratio_low': ready_low,
                'ready_stance_ratio_high': ready_high,
                'ideal_ready_stance_ratio': guidance['ideal_ready_stance_ratio'],
                'ideal_knee_angle': guidance['ideal_knee_angle'],
                'recovery_ratio_target': guidance['recovery_ratio_target']
            }
        }

    def process_video_frame(self, frame: np.ndarray, timestamp: float, frame_count: int = 0) -> Tuple[
        np.ndarray, Optional[Dict]]:
        """Highly optimized video frame processing with robust error handling"""
        if frame is None:
            return np.zeros((300, 400, 3), dtype=np.uint8), None

        try:
            h, w = frame.shape[:2]
            
            # Use frame reference instead of copying when possible
            person_roi_frame = frame
            roi_offset = np.array([0, 0], dtype=np.int32)

            # Optimized ROI calculation for person tracking
            if self.tracked_id is not None and self.prev_person_box is not None:
                try:
                    x1, y1, x2, y2 = self.prev_person_box.astype(np.int32)
                    
                    # Fast padding calculation
                    width = x2 - x1
                    height = y2 - y1
                    pad_x = width * 2  # 200% expansion, simplified
                    pad_y = height * 2
                    
                    # Boundary checks with fast min/max
                    roi_x1 = max(0, int(x1 - pad_x))
                    roi_y1 = max(0, int(y1 - pad_y))
                    roi_x2 = min(w, int(x2 + pad_x))
                    roi_y2 = min(h, int(y2 + pad_y))

                    # Only use ROI if it's significantly smaller than full frame
                    if (roi_x2 - roi_x1) * (roi_y2 - roi_y1) < 0.8 * w * h:
                        person_roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                        roi_offset[0] = roi_x1
                        roi_offset[1] = roi_y1
                except Exception as e:
                    print(f"ROI calculation error: {e}")
                    # Use full frame as fallback
                    person_roi_frame = frame
                    roi_offset = np.array([0, 0], dtype=np.int32)
            
            # Optimized pose detection
            try:
                with torch.no_grad():
                    pose_results = self.pose_model(person_roi_frame, conf=self.conf_threshold)[0]

                # Fast keypoint processing
                keypoints_data = pose_results.keypoints.data.cpu()
                if roi_offset[0] != 0 or roi_offset[1] != 0:
                    # Vectorized coordinate adjustment
                    keypoints_data[:, :, :2] += torch.from_numpy(roi_offset).float().unsqueeze(0)

                if not self.validate_keypoints(keypoints_data):
                    return frame, None

                # Update tracking with optimized operations
                self.update_tracked_player(pose_results)

                # Validate data before analysis
                if not self.validate_frame_data(frame, keypoints_data):
                    return frame, None

                return self.analyze_frame(frame, keypoints_data, timestamp)
                
            except Exception as e:
                print(f"Pose detection error: {e}")
                return frame, None

        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            # Reset tracking if we're having persistent errors
            if hasattr(self, '_error_count'):
                self._error_count += 1
                if self._error_count > 10:  # Reset after 10 consecutive errors
                    self.reset_tracking()
                    self._error_count = 0
            else:
                self._error_count = 1
            return frame, None

    def validate_keypoints(self, keypoints_data: torch.Tensor) -> bool:
        """Fast keypoints validation with minimal checks"""
        return (keypoints_data.numel() > 0 and 
                keypoints_data.shape[1] > self.KEYPOINTS['RIGHT_ANKLE'])

    def update_tracked_player(self, pose_results: Any) -> None:
        """Optimized player tracking with minimal CPU transfers"""
        keypoints_data = pose_results.keypoints.data
        
        # Only transfer to CPU once we need the data
        if self.tracked_id is None:
            # Fast player selection based on wrist confidence
            valid_players = []
            keypoints_cpu = keypoints_data.cpu()
            
            for i, kp in enumerate(keypoints_cpu):
                if kp.shape[0] > self.KEYPOINTS['RIGHT_WRIST']:
                    try:
                        conf = kp[self.KEYPOINTS['RIGHT_WRIST'], 2].item()
                        if conf > 0.3:  # Quick confidence threshold
                            valid_players.append((i, conf))
                    except (IndexError, RuntimeError):
                        continue

            if valid_players:
                self.tracked_id = max(valid_players, key=lambda x: x[1])[0]
                # Get bounding box
                if hasattr(pose_results, 'boxes') and pose_results.boxes is not None:
                    boxes_data = pose_results.boxes.xyxy.cpu().numpy()
                    self.prev_person_box = boxes_data[self.tracked_id]
                    
        elif self.tracked_id >= len(keypoints_data):
            # Reset tracking if ID is invalid
            self.tracked_id = None
            self.prev_person_box = None
        else:
            # Update bounding box for existing tracked player
            if hasattr(pose_results, 'boxes') and pose_results.boxes is not None:
                boxes_data = pose_results.boxes.xyxy.cpu().numpy()
                if self.tracked_id < len(boxes_data):
                    self.prev_person_box = boxes_data[self.tracked_id]

    def analyze_frame(self, frame: np.ndarray, keypoints_data: torch.Tensor, timestamp: float) -> Tuple[
        np.ndarray, Optional[Dict]]:
        """Highly optimized frame analysis with minimal copying and robust error handling"""
        try:
            if self.tracked_id is None or self.tracked_id >= len(keypoints_data):
                self.tracked_id = None
                return frame, None

            # Get keypoints and validate
            try:
                current_keypoints = keypoints_data[self.tracked_id].numpy()
                if current_keypoints.shape[0] <= self.KEYPOINTS['RIGHT_WRIST']:
                    return frame, None
            except Exception as e:
                print(f"Keypoints extraction error: {e}")
                return frame, None

            # Optimized keypoint smoothing with smaller buffer
            self.keypoint_buffer.append(current_keypoints)
            if len(self.keypoint_buffer) > self.buffer_size:
                self.keypoint_buffer.pop(0)

            # Fast keypoint averaging
            if len(self.keypoint_buffer) > 1:
                keypoints = np.mean(self.keypoint_buffer, axis=0)
            else:
                keypoints = current_keypoints

            # Optimized distance calculation
            try:
                wrist_point = keypoints[self.KEYPOINTS['RIGHT_WRIST'], :2]
                
                # Check cache first for shuttlecock detection
                shuttle_box = self.detect_shuttlecock(frame)
                current_distance = calculate_shuttle_distance_squared(shuttle_box, wrist_point) if shuttle_box is not None else float('inf')
            except Exception as e:
                print(f"Distance calculation error: {e}")
                wrist_point = np.array([0, 0])
                shuttle_box = None
                current_distance = float('inf')

            # Calculate angles
            try:
                grip_angle = self.calculate_grip_angle(keypoints)
                shoulder_angle = self.calculate_shoulder_angle(keypoints)
            except Exception as e:
                print(f"Angle calculation error: {e}")
                grip_angle = 0.0
                shoulder_angle = 0.0

            # Optimized minimum distance tracking
            try:
                distance_threshold_sq = 25.0  # 5^2 for squared distance
                if current_distance < self.min_shuttle_distance - distance_threshold_sq and shuttle_box is not None:
                    self.min_shuttle_distance = current_distance
                    # Use references instead of copies where possible
                    self.min_distance_frame = {
                        'frame': frame,  # Reference, not copy
                        'keypoints': keypoints,
                        'timestamp': timestamp,
                        'shuttle_box': shuttle_box
                    }
            except Exception as e:
                print(f"Distance tracking error: {e}")

            # Fast phase determination
            try:
                shot_phase = self.determine_shot_phase_improved(
                    math.sqrt(current_distance) if current_distance != float('inf') else float('inf'),
                    grip_angle,
                    shoulder_angle
                )
            except Exception as e:
                print(f"Phase determination error: {e}")
                shot_phase = None

            # Optimized tracking state update
            try:
                self.update_tracking_state(frame, keypoints, shot_phase, timestamp, current_distance)
            except Exception as e:
                print(f"Tracking state update error: {e}")

            # Fast analysis object creation
            try:
                analysis = FrameAnalysis(
                    timestamp=timestamp,
                    grip_angle=grip_angle,
                    shoulder_angle=shoulder_angle,
                    shot_type=self.target_shot_type,
                    shot_phase=shot_phase,
                    within_threshold=shot_phase is not None,
                    keypoints=keypoints
                )

                self.last_analysis = analysis
                self.update_key_frames_from_detections()

                # Use to_dict() method since __slots__ removes __dict__
                try:
                    analysis_dict = analysis.to_dict()
                except Exception as e:
                    print(f"Error converting analysis to dict: {e}")
                    analysis_dict = {
                        'timestamp': timestamp,
                        'grip_angle': grip_angle,
                        'shoulder_angle': shoulder_angle,
                        'shot_type': self.target_shot_type,
                        'shot_phase': shot_phase,
                        'within_threshold': shot_phase is not None,
                        'keypoints': keypoints.tolist() if isinstance(keypoints, np.ndarray) else []
                    }
                
                return self.visualize_analysis(frame, analysis, shuttle_box), analysis_dict
                
            except Exception as e:
                print(f"Analysis object creation error: {e}")
                return frame, None
                
        except Exception as e:
            print(f"General frame analysis error: {e}")
            return frame, None

    def determine_shot_phase_improved(self, current_distance: float, grip_angle: float,
                               shoulder_angle: float) -> Optional[str]:
        """Highly optimized phase detection with cached threshold checks"""
        thresholds = self._thresholds_cache
        if not thresholds:
            return None

        # Pre-compute threshold checks to avoid repeated calculations
        pre_hit_thresh = thresholds.get('pre_hit', {})
        hit_thresh = thresholds.get('hit', {})
        follow_thresh = thresholds.get('follow_through', {})
        
        if not all([pre_hit_thresh, hit_thresh, follow_thresh]):
            return None

        # Fast threshold checks using cached function
        in_pre_hit = is_within_threshold(grip_angle, shoulder_angle, pre_hit_thresh)
        in_hit = is_within_threshold(grip_angle, shoulder_angle, hit_thresh)
        in_follow = is_within_threshold(grip_angle, shoulder_angle, follow_thresh)

        # Optimized state machine with early returns
        if self.sequence_state == 'initial':
            if in_pre_hit:
                self.sequence_state = 'pre_hit'
                return 'pre_hit'

        elif self.sequence_state == 'pre_hit':
            if in_hit:
                self.sequence_state = 'looking_for_hit'
                self.found_hit = True
                self.min_shuttle_distance = current_distance * current_distance  # Store squared distance
                self.min_distance_frame = None
                return 'hit'
            elif in_pre_hit:
                return 'pre_hit'
            elif in_follow:
                self.sequence_state = 'looking_for_hit'
                self.awaiting_follow_through = True
                return 'pre_hit'

        elif self.sequence_state == 'looking_for_hit':
            if not self.found_hit and in_hit:
                self.found_hit = True
                return 'hit'
            elif in_follow and (self.found_hit or self.awaiting_follow_through):
                self.sequence_state = 'looking_for_follow'
                self.awaiting_follow_through = False
                return 'follow_through'
            elif self.found_hit:
                return 'hit'
            else:
                return 'pre_hit'

        elif self.sequence_state == 'looking_for_follow':
            if in_follow:
                return 'follow_through'
            elif in_pre_hit:
                self.reset_detection_state()
                self.sequence_state = 'pre_hit'
                return 'pre_hit'

        return None

    def reset_detection_state(self):
        """Fast detection state reset"""
        self.sequence_state = 'initial'
        self.found_hit = False
        self.awaiting_follow_through = False
        self.min_shuttle_distance = float('inf')
        self.min_distance_frame = None
        # Clear phase detections efficiently
        for phase in self.phase_detections:
            self.phase_detections[phase].clear()

    def reset_tracking(self):
        """Reset all tracking state to recover from errors"""
        self.tracked_id = None
        self.prev_person_box = None
        self.prev_shuttle_box = None
        self.keypoint_buffer.clear()
        self.sequence_state = 'initial'
        self.found_hit = False
        self.awaiting_follow_through = False
        self.min_shuttle_distance = float('inf')
        self.min_distance_frame = None
        self._error_count = 0
        
        # Clear caches
        self.clear_cache()
        
        print("Tracking state reset due to errors")

    def update_tracking_state(self, frame: np.ndarray, keypoints: np.ndarray, phase: Optional[str], 
                             timestamp: float, current_distance: float) -> None:
        """Optimized tracking state update with minimal data copying"""
        if phase:
            # Store only essential data, avoid frame copying
            detection_data = {
                'frame': frame,  # Use reference instead of copy
                'keypoints': keypoints,
                'timestamp': timestamp,
                'distance': current_distance
            }
            self.phase_detections[phase].append(detection_data)
            
            # Limit detection history to prevent memory bloat
            if len(self.phase_detections[phase]) > 10:
                self.phase_detections[phase].pop(0)

        # Update state variables
        self.last_shuttle_distance = current_distance
        self.prev_keypoints = keypoints
        self.prev_time = timestamp

    def visualize_analysis(self, frame: np.ndarray, analysis: FrameAnalysis,
                          shuttle_box: Optional[np.ndarray]) -> np.ndarray:
        """Modern visualization with enhanced graphics"""
        viz_frame = frame.copy()
        keypoints = analysis.keypoints

        # Define skeleton connections
        connections = [
            (self.KEYPOINTS['LEFT_SHOULDER'], self.KEYPOINTS['RIGHT_SHOULDER']),
            (self.KEYPOINTS['RIGHT_SHOULDER'], self.KEYPOINTS['RIGHT_ELBOW']),
            (self.KEYPOINTS['RIGHT_ELBOW'], self.KEYPOINTS['RIGHT_WRIST']),
            (self.KEYPOINTS['LEFT_SHOULDER'], self.KEYPOINTS['LEFT_HIP']),
            (self.KEYPOINTS['RIGHT_SHOULDER'], self.KEYPOINTS['RIGHT_HIP']),
            (self.KEYPOINTS['LEFT_HIP'], self.KEYPOINTS['RIGHT_HIP']),
            (self.KEYPOINTS['LEFT_HIP'], self.KEYPOINTS['LEFT_KNEE']),
            (self.KEYPOINTS['RIGHT_HIP'], self.KEYPOINTS['RIGHT_KNEE']),
            (self.KEYPOINTS['LEFT_KNEE'], self.KEYPOINTS['LEFT_ANKLE']),
            (self.KEYPOINTS['RIGHT_KNEE'], self.KEYPOINTS['RIGHT_ANKLE'])
        ]

        # Get phase for styling
        phase = analysis.shot_phase or 'default'
        footwork_guidance = self.FOOTWORK_GUIDANCE.get(self.target_shot_type, self.FOOTWORK_GUIDANCE['default'])
        footwork_metrics = self._calculate_footwork_phase_metrics(keypoints, phase)

        # Draw modern skeleton
        draw_modern_skeleton(viz_frame, keypoints, connections, (0, 0, 0), phase)
        
        # Draw modern shuttlecock
        draw_modern_shuttlecock(viz_frame, shuttle_box)

        # Draw lower-body footwork tracking
        draw_modern_footwork_tracking(viz_frame, keypoints, footwork_metrics, footwork_guidance, phase)
        
        # Draw modern info overlay
        draw_modern_info_overlay(viz_frame, analysis, footwork_metrics, footwork_guidance)

        # Draw connection line between wrist and shuttlecock
        if shuttle_box is not None:
            wrist_point = keypoints[self.KEYPOINTS['RIGHT_WRIST'], :2]
            current_distance = calculate_shuttle_distance(shuttle_box, wrist_point)
            
            center_x = int((shuttle_box[0] + shuttle_box[2]) * 0.5)
            center_y = int((shuttle_box[1] + shuttle_box[3]) * 0.5)
            
            draw_modern_connection_line(
                viz_frame,
                (int(wrist_point[0]), int(wrist_point[1])),
                (center_x, center_y),
                current_distance
            )

        return viz_frame

    def get_technique_summary(self) -> Dict:
        """Highly optimized technique summary generation"""
        # Quick completeness check
        if not all(self.key_frames.values()):
            return {
                'status': 'incomplete',
                'message': 'Not all key frames captured'
            }

        try:
            # Pre-allocate dictionaries for better performance
            serializable_key_frames = {}
            timing_data = {}
            calculated_angles = {}
            ideal_angles_data = {}
            angle_differences = {}
            footwork_phase_metrics = {}
            
            # Cache shot thresholds
            shot_thresholds = self.SHOT_THRESHOLDS.get(self.target_shot_type, {})
            
            # Process key frames efficiently
            for phase, data in self.key_frames.items():
                if data:
                    keypoints_val = data.get('keypoints')
                    shuttle_box_val = data.get('shuttle_box')
                    distance_val = data.get('distance')
                    
                    serializable_data = {
                        'timestamp': data.get('timestamp'),
                        'keypoints': keypoints_val.tolist() if isinstance(keypoints_val, np.ndarray) else keypoints_val,
                        'shuttle_box': shuttle_box_val.tolist() if isinstance(shuttle_box_val, np.ndarray) else shuttle_box_val,
                        'distance': distance_val if isinstance(distance_val, (int, float)) and not math.isnan(distance_val) and not math.isinf(distance_val) else None
                    }
                    serializable_key_frames[phase] = serializable_data
                else:
                    serializable_key_frames[phase] = None

            # Fast timestamp extraction and validation
            timestamps = {}
            for phase in ['pre_hit', 'hit', 'follow_through']:
                frame_data = serializable_key_frames.get(phase)
                if frame_data:
                    ts = frame_data.get('timestamp')
                    if isinstance(ts, (int, float)) and not math.isnan(ts) and not math.isinf(ts):
                        timestamps[phase] = float(ts)

            # Calculate timing efficiently
            if 'hit' in timestamps and 'pre_hit' in timestamps:
                prep_time = timestamps['hit'] - timestamps['pre_hit']
                if 0 <= prep_time < 60.0:
                    timing_data['preparation_time'] = prep_time
                    
            if 'follow_through' in timestamps and 'hit' in timestamps:
                follow_time = timestamps['follow_through'] - timestamps['hit']
                if 0 <= follow_time < 60.0:
                    timing_data['follow_through_time'] = follow_time

            # Optimized angle calculations
            for phase, data in self.key_frames.items():
                calculated_angle = None
                
                if data and 'keypoints' in data:
                    keypoints_data = data['keypoints']
                    
                    # Convert to numpy array if needed
                    if isinstance(keypoints_data, list):
                        keypoints = np.array(keypoints_data)
                    else:
                        keypoints = keypoints_data
                        
                    # Validate keypoints
                    if (isinstance(keypoints, np.ndarray) and 
                        keypoints.shape[0] > self.KEYPOINTS['RIGHT_WRIST'] and
                        len(keypoints.shape) >= 2 and keypoints.shape[1] >= 3):
                        
                        # Calculate angle directly
                        calculated_angle = self.calculate_grip_angle(keypoints)
                            
                        # Validate angle
                        if (isinstance(calculated_angle, (int, float)) and 
                            not math.isnan(calculated_angle) and 
                            not math.isinf(calculated_angle) and
                            0 <= calculated_angle <= 180):
                            pass  # Valid angle
                        else:
                            calculated_angle = None

                calculated_angles[phase] = calculated_angle

                # Get ideal angles efficiently
                phase_thresholds = shot_thresholds.get(phase, {})
                ideal_angle = phase_thresholds.get('grip')
                
                if (isinstance(ideal_angle, (int, float)) and 
                    not math.isnan(ideal_angle) and 
                    not math.isinf(ideal_angle)):
                    ideal_angles_data[phase] = float(ideal_angle)
                else:
                    ideal_angles_data[phase] = None

                # Calculate differences
                if (isinstance(calculated_angle, (int, float)) and 
                    isinstance(ideal_angles_data[phase], (int, float))):
                    angle_differences[phase] = calculated_angle - ideal_angles_data[phase]
                else:
                    angle_differences[phase] = None

                if data and 'keypoints' in data:
                    keypoints_data = data['keypoints']
                    if isinstance(keypoints_data, list):
                        keypoints = np.array(keypoints_data)
                    else:
                        keypoints = keypoints_data

                    if (isinstance(keypoints, np.ndarray) and len(keypoints.shape) >= 2 and
                            keypoints.shape[0] > self.KEYPOINTS['RIGHT_ANKLE']):
                        footwork_phase_metrics[phase] = self._calculate_footwork_phase_metrics(keypoints, phase)
                    else:
                        footwork_phase_metrics[phase] = {
                            'phase': phase,
                            'stance_width': None,
                            'stance_width_ratio': None,
                            'shoulder_width': None,
                            'hip_width': None,
                            'ankle_span_x': None,
                            'ankle_span_y': None,
                            'left_knee_angle': None,
                            'right_knee_angle': None,
                            'average_knee_angle': None,
                            'hip_center': None,
                            'ankle_center': None,
                            'phase_note': '',
                            'flags': [],
                            'ideal': {
                                'ready_stance_ratio': None,
                                'knee_angle': None,
                                'recovery_ratio_target': None
                            }
                        }
                else:
                    footwork_phase_metrics[phase] = {
                        'phase': phase,
                        'stance_width': None,
                        'stance_width_ratio': None,
                        'shoulder_width': None,
                        'hip_width': None,
                        'ankle_span_x': None,
                        'ankle_span_y': None,
                        'left_knee_angle': None,
                        'right_knee_angle': None,
                        'average_knee_angle': None,
                        'hip_center': None,
                        'ankle_center': None,
                        'phase_note': '',
                        'flags': [],
                        'ideal': {
                            'ready_stance_ratio': None,
                            'knee_angle': None,
                            'recovery_ratio_target': None
                        }
                    }

            footwork_summary = self._build_footwork_summary(footwork_phase_metrics)

            return {
                'status': 'complete',
                'shot_type': self.target_shot_type,
                'key_frames': serializable_key_frames,
                'timing': timing_data,
                'angles': calculated_angles,
                'ideal_angles': ideal_angles_data,
                'angle_differences': angle_differences,
                'footwork': footwork_summary
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to generate technique summary: {str(e)}'
            }

    def update_key_frames_from_detections(self) -> None:
        """Optimized key frames update based on phase detections"""
        # Pre-hit: Get the middle frame between first and last detection
        if len(self.phase_detections['pre_hit']) > 1:
            middle_idx = len(self.phase_detections['pre_hit']) // 2
            self.key_frames['pre_hit'] = self.phase_detections['pre_hit'][middle_idx]
        elif len(self.phase_detections['pre_hit']) == 1:
            self.key_frames['pre_hit'] = self.phase_detections['pre_hit'][0]

        # Hit: Use the frame with minimum distance between wrist and shuttlecock
        if self.min_distance_frame is not None:
            self.key_frames['hit'] = self.min_distance_frame
        elif len(self.phase_detections['hit']) > 0:
            # Find the frame with minimum distance in hit phase detections
            min_dist_idx = 0
            min_dist = float('inf')
            for i, detection in enumerate(self.phase_detections['hit']):
                if detection.get('distance', float('inf')) < min_dist:
                    min_dist = detection['distance']
                    min_dist_idx = i
            self.key_frames['hit'] = self.phase_detections['hit'][min_dist_idx]

        # Follow-through: Get the last frame
        if len(self.phase_detections['follow_through']) > 0:
            self.key_frames['follow_through'] = self.phase_detections['follow_through'][-1]

    def clear_cache(self):
        """Clear all caches to free memory"""
        _cached_threshold_check.cache_clear()
        self._roi_cache.clear()
        
    def _analysis_to_dict(self, analysis: Optional[FrameAnalysis]) -> Optional[Dict]:
        """Safely convert FrameAnalysis to dictionary"""
        if analysis is None:
            return None
        
        try:
            return analysis.to_dict()
        except Exception as e:
            print(f"Error converting analysis to dict: {e}")
            # Return a safe fallback dictionary
            return {
                'timestamp': getattr(analysis, 'timestamp', 0.0),
                'grip_angle': getattr(analysis, 'grip_angle', 0.0),
                'shoulder_angle': getattr(analysis, 'shoulder_angle', 0.0),
                'shot_type': getattr(analysis, 'shot_type', 'unknown'),
                'shot_phase': getattr(analysis, 'shot_phase', None),
                'within_threshold': getattr(analysis, 'within_threshold', False),
                'keypoints': []
            }
