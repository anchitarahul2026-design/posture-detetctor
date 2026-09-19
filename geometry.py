# geometry.py - Enhanced with comprehensive body measurements

import math
from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import statistics


def calculate_angle(a, b, c) -> float:
    """
    Generic mathematical utility to calculate joint angle (in degrees) 
    at vertex 'b' given points a, b, and c.
    """
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def distance(a, b) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def horizontal_distance(a, b) -> float:
    """Calculate horizontal (x-axis) distance between two points."""
    return abs(a.x - b.x)


def vertical_distance(a, b) -> float:
    """Calculate vertical (y-axis) distance between two points."""
    return abs(a.y - b.y)


def midpoint(a, b):
    """Calculate the midpoint between two points."""
    return type(a)(x=(a.x + b.x)/2, y=(a.y + b.y)/2, z=(a.z + b.z)/2)


def angle_between_vectors(v1, v2) -> float:
    """Calculate the angle between two vectors in degrees."""
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def get_vector(p1, p2):
    """Get vector from p1 to p2."""
    return (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)


def is_landmark_valid(landmark, confidence_threshold=0.5) -> bool:
    """
    Check if a landmark has sufficient visibility and confidence.
    MediaPipe landmarks have a 'visibility' attribute (0-1).
    """
    if not landmark:
        return False
    # Check if landmark has visibility attribute
    if hasattr(landmark, 'visibility'):
        return landmark.visibility >= confidence_threshold
    # Fallback: check if coordinates are reasonable
    return 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1


def get_landmark_confidence(landmark) -> float:
    """Get the confidence/visibility of a landmark."""
    if hasattr(landmark, 'visibility'):
        return landmark.visibility
    return 1.0  # Default if no visibility


# ============================================================================
# HEAD MEASUREMENTS
# ============================================================================

def calculate_head_forward_ratio(nose, left_shoulder, right_shoulder) -> Optional[float]:
    """
    Calculate head forward displacement normalized by shoulder width.
    Positive values indicate forward head position.
    """
    shoulder_width = distance(left_shoulder, right_shoulder)
    if shoulder_width < 0.01:
        return None
    
    # Use the vertical line between shoulders as reference
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    
    # Horizontal displacement from shoulder center to nose
    forward_displacement = nose.x - shoulder_center.x
    
    # Normalize by shoulder width
    return forward_displacement / shoulder_width


def calculate_head_tilt(nose, left_ear, right_ear) -> Optional[float]:
    """
    Calculate head tilt angle in degrees.
    Positive = tilted to the right, negative = tilted to the left.
    """
    if not left_ear or not right_ear:
        return None
    
    # Vector between ears
    ear_vector = get_vector(left_ear, right_ear)
    # Horizontal reference vector
    horizontal = (1.0, 0.0, 0.0)
    
    # Angle between ear line and horizontal
    angle = angle_between_vectors(ear_vector, horizontal)
    
    # Determine sign (tilt direction)
    if right_ear.y > left_ear.y:  # Right ear lower = tilted right
        return angle
    else:
        return -angle


def calculate_head_rotation(nose, left_shoulder, right_shoulder) -> Optional[float]:
    """
    Estimate head rotation based on nose position relative to shoulders.
    Returns rotation in degrees (-90 to 90), negative = looking left.
    """
    shoulder_width = distance(left_shoulder, right_shoulder)
    if shoulder_width < 0.01:
        return None
    
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    nose_offset = nose.x - shoulder_center_x
    
    # Map to angle: full shoulder width = ~90 degrees of head rotation
    rotation = (nose_offset / shoulder_width) * 90.0
    return max(-90.0, min(90.0, rotation))


# ============================================================================
# SHOULDER MEASUREMENTS
# ============================================================================

def calculate_shoulder_tilt(left_shoulder, right_shoulder) -> float:
    """
    Calculate shoulder tilt angle in degrees.
    Positive = right shoulder higher, negative = left shoulder higher.
    """
    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    
    if abs(dx) < 0.001:
        return 90.0 if dy > 0 else -90.0
    
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def calculate_shoulder_width(left_shoulder, right_shoulder) -> float:
    """Calculate shoulder width (horizontal distance)."""
    return horizontal_distance(left_shoulder, right_shoulder)


def calculate_shoulder_symmetry(left_shoulder, right_shoulder, 
                               left_hip, right_hip) -> Optional[float]:
    """
    Calculate shoulder symmetry relative to hips.
    Returns a score: 0 = perfectly symmetric, higher = more asymmetric.
    """
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    
    # Vertical displacement from hip center to shoulder center
    vertical_offset = shoulder_center.y - hip_center.y
    
    # Normalize by shoulder width
    shoulder_width = distance(left_shoulder, right_shoulder)
    if shoulder_width < 0.01:
        return None
    
    # How much the shoulders deviate from horizontal
    tilt = calculate_shoulder_tilt(left_shoulder, right_shoulder)
    
    # Combine: tilt magnitude + lateral offset
    lateral_offset = abs(shoulder_center.x - hip_center.x) / shoulder_width
    tilt_magnitude = abs(tilt) / 45.0  # Normalize to 0-1 range (45 deg max)
    
    symmetry_score = (lateral_offset * 0.5 + tilt_magnitude * 0.5)
    return min(1.0, symmetry_score)


# ============================================================================
# TORSO MEASUREMENTS
# ============================================================================

def calculate_torso_angle(shoulder_center, hip_center, vertical_reference=None) -> float:
    """
    Calculate torso angle relative to vertical.
    Returns degrees of lean from vertical.
    """
    if vertical_reference is None:
        vertical = (0.0, 1.0, 0.0)  # Y-up
    else:
        vertical = vertical_reference
    
    torso_vector = get_vector(hip_center, shoulder_center)
    angle = angle_between_vectors(torso_vector, vertical)
    return angle


def calculate_torso_lateral_lean(shoulder_center, hip_center) -> float:
    """
    Calculate lateral lean of torso (left/right).
    Returns positive = leaning right, negative = leaning left.
    """
    # Horizontal displacement normalized by shoulder width
    dx = shoulder_center.x - hip_center.x
    return dx


def calculate_torso_forward_lean(nose, shoulder_center, hip_center) -> Optional[float]:
    """
    Calculate forward/backward lean using nose, shoulders, and hips.
    Returns positive = leaning forward.
    """
    # Vertical distance from shoulders to hips
    torso_height = abs(shoulder_center.y - hip_center.y)
    if torso_height < 0.01:
        return None
    
    # Horizontal displacement of nose relative to shoulder-hip line
    # Using the line from hips to shoulders as reference
    shoulder_y = shoulder_center.y
    hip_y = hip_center.y
    
    # The "forward" direction in the image plane
    # For a neutral posture, nose should be roughly above shoulder center
    nose_offset = nose.x - shoulder_center.x
    
    # Normalize by torso height
    return nose_offset / torso_height


# ============================================================================
# HIP MEASUREMENTS
# ============================================================================

def calculate_hip_tilt(left_hip, right_hip) -> float:
    """
    Calculate hip tilt angle in degrees.
    Positive = right hip higher, negative = left hip higher.
    """
    dx = right_hip.x - left_hip.x
    dy = right_hip.y - left_hip.y
    
    if abs(dx) < 0.001:
        return 90.0 if dy > 0 else -90.0
    
    angle = math.degrees(math.atan2(dy, dx))
    return angle


def calculate_hip_width(left_hip, right_hip) -> float:
    """Calculate hip width (horizontal distance between hips)."""
    return horizontal_distance(left_hip, right_hip)


def calculate_hip_symmetry(left_hip, right_hip, 
                          left_shoulder, right_shoulder) -> Optional[float]:
    """
    Calculate hip symmetry relative to shoulders.
    """
    hip_center = midpoint(left_hip, right_hip)
    shoulder_center = midpoint(left_shoulder, right_shoulder)
    
    hip_width = distance(left_hip, right_hip)
    if hip_width < 0.01:
        return None
    
    # Lateral offset of hips relative to shoulders
    lateral_offset = abs(hip_center.x - shoulder_center.x) / hip_width
    
    # Hip tilt magnitude
    tilt = abs(calculate_hip_tilt(left_hip, right_hip))
    tilt_magnitude = tilt / 45.0
    
    symmetry_score = (lateral_offset * 0.5 + tilt_magnitude * 0.5)
    return min(1.0, symmetry_score)


# ============================================================================
# LEG MEASUREMENTS
# ============================================================================

def calculate_knee_angle(hip, knee, ankle) -> float:
    """
    Calculate knee angle in degrees.
    Uses hip, knee, and ankle landmarks.
    """
    return calculate_angle(hip, knee, ankle)


def calculate_stance_width(left_hip, right_hip) -> float:
    """Calculate stance width (horizontal distance between hips)."""
    return horizontal_distance(left_hip, right_hip)


def calculate_leg_symmetry(left_knee_angle, right_knee_angle) -> float:
    """
    Calculate leg symmetry based on knee angles.
    Returns 0 = perfectly symmetric, higher = more asymmetric.
    """
    if left_knee_angle is None or right_knee_angle is None:
        return 0.0
    
    diff = abs(left_knee_angle - right_knee_angle)
    # Normalize: 20 degree difference = 1.0
    return min(1.0, diff / 20.0)


# ============================================================================
# COMBINED MEASUREMENTS
# ============================================================================

def extract_all_body_measurements(landmarks, confidence_threshold=0.5):
    """
    Extract body measurements robustly.

    IMPORTANT:
    Hips are optional because a seated user may have their hips
    outside the webcam frame.
    """

    if not landmarks:
        return {"valid": False}

    def get_landmark(index):
        if index < len(landmarks):
            return landmarks[index]
        return None

    # ---------------------------------------------------------
    # REQUIRED FOR CALIBRATION
    # ---------------------------------------------------------
    nose = get_landmark(0)
    left_shoulder = get_landmark(11)
    right_shoulder = get_landmark(12)

    required = [
        (0, nose),
        (11, left_shoulder),
        (12, right_shoulder),
    ]

    for index, landmark in required:
        if not is_landmark_valid(
            landmark,
            confidence_threshold
        ):
            return {
                "valid": False,
                "missing_landmark": index
            }

    # ---------------------------------------------------------
    # OPTIONAL LANDMARKS
    # ---------------------------------------------------------
    left_ear = get_landmark(7)
    right_ear = get_landmark(8)

    left_hip = get_landmark(23)
    right_hip = get_landmark(24)

    left_knee = get_landmark(25)
    right_knee = get_landmark(26)

    left_ankle = get_landmark(27)
    right_ankle = get_landmark(28)

    # ---------------------------------------------------------
    # BASIC MEASUREMENTS
    # ---------------------------------------------------------
    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    measurements = {
        "valid": True,

        "confidences": {
            "nose": get_landmark_confidence(nose),
            "left_shoulder": get_landmark_confidence(
                left_shoulder
            ),
            "right_shoulder": get_landmark_confidence(
                right_shoulder
            ),
        },

        "shoulder_center": shoulder_center,
    }

    # ---------------------------------------------------------
    # HEAD
    # ---------------------------------------------------------
    head_forward = calculate_head_forward_ratio(
        nose,
        left_shoulder,
        right_shoulder
    )

    if head_forward is not None:
        measurements["head_forward_ratio"] = head_forward

    head_tilt = calculate_head_tilt(
        nose,
        left_ear,
        right_ear
    )

    if head_tilt is not None:
        measurements["head_tilt"] = head_tilt

    head_rotation = calculate_head_rotation(
        nose,
        left_shoulder,
        right_shoulder
    )

    if head_rotation is not None:
        measurements["head_rotation"] = head_rotation

    # ---------------------------------------------------------
    # SHOULDERS
    # ---------------------------------------------------------
    measurements["shoulder_tilt"] = calculate_shoulder_tilt(
        left_shoulder,
        right_shoulder
    )

    measurements["shoulder_width"] = calculate_shoulder_width(
        left_shoulder,
        right_shoulder
    )

    # ---------------------------------------------------------
    # LOWER BODY IS OPTIONAL
    # ---------------------------------------------------------
    hips_available = (
        is_landmark_valid(
            left_hip,
            confidence_threshold
        )
        and
        is_landmark_valid(
            right_hip,
            confidence_threshold
        )
    )

    if hips_available:

        hip_center = midpoint(
            left_hip,
            right_hip
        )

        measurements["hip_center"] = hip_center

        shoulder_symmetry = calculate_shoulder_symmetry(
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
        )

        if shoulder_symmetry is not None:
            measurements["shoulder_symmetry"] = (
                shoulder_symmetry
            )

        measurements["torso_angle"] = calculate_torso_angle(
            shoulder_center,
            hip_center
        )

        measurements["torso_lateral_lean"] = (
            calculate_torso_lateral_lean(
                shoulder_center,
                hip_center
            )
        )

        torso_forward = calculate_torso_forward_lean(
            nose,
            shoulder_center,
            hip_center
        )

        if torso_forward is not None:
            measurements["torso_forward_lean"] = (
                torso_forward
            )

        measurements["hip_tilt"] = calculate_hip_tilt(
            left_hip,
            right_hip
        )

        measurements["hip_width"] = calculate_hip_width(
            left_hip,
            right_hip
        )

        hip_symmetry = calculate_hip_symmetry(
            left_hip,
            right_hip,
            left_shoulder,
            right_shoulder
        )

        if hip_symmetry is not None:
            measurements["hip_symmetry"] = hip_symmetry

        if (
            left_knee
            and left_ankle
            and is_landmark_valid(
                left_knee,
                confidence_threshold
            )
            and is_landmark_valid(
                left_ankle,
                confidence_threshold
            )
        ):
            measurements["left_knee_angle"] = (
                calculate_knee_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )
            )

        if (
            right_knee
            and right_ankle
            and is_landmark_valid(
                right_knee,
                confidence_threshold
            )
            and is_landmark_valid(
                right_ankle,
                confidence_threshold
            )
        ):
            measurements["right_knee_angle"] = (
                calculate_knee_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )
            )

        measurements["stance_width"] = (
            calculate_stance_width(
                left_hip,
                right_hip
            )
        )

        if (
            "left_knee_angle" in measurements
            and "right_knee_angle" in measurements
        ):
            measurements["leg_symmetry"] = (
                calculate_leg_symmetry(
                    measurements["left_knee_angle"],
                    measurements["right_knee_angle"]
                )
            )

    return measurements


# ============================================================================
# EXISTING CLASSES (Preserved)
# ============================================================================

class PostureAnalyzer:
    def __init__(self, baseline_threshold=0.85):
        self.baseline_ratio = None
        self.threshold = baseline_threshold

    def compute_ratio(self, nose, left_shoulder, right_shoulder):
        """
        Computes scale-invariant posture ratio:
        Ratio = (Nose-to-Shoulder-Midpoint Vertical Distance) / (Shoulder-to-Shoulder Width)
        """
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        neck_length = abs(mid_shoulder_y - nose.y)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        if shoulder_width == 0:
            return None

        return neck_length / shoulder_width

    def calibrate(self, ratio):
        self.baseline_ratio = ratio

    def is_misaligned(self, current_ratio):
        if self.baseline_ratio is None or current_ratio is None:
            return False
        return current_ratio < (self.baseline_ratio * self.threshold)


class HandAnalyzer:
    def __init__(self, history_length=5):
        self.history_length = history_length
        self.angle_history = {}

        # Landmark indices for MCP, PIP, DIP joints per finger
        self.joint_map = {
            "THUMB": (1, 2, 3),
            "INDEX": (5, 6, 7),
            "MIDDLE": (9, 10, 11),
            "RING": (13, 14, 15),
            "PINKY": (17, 18, 19)
        }

    def _get_finger_angles(self, hand_landmarks) -> dict:
        landmarks = hand_landmarks.landmark
        angles = {}

        for finger_name, (mcp_idx, pip_idx, dip_idx) in self.joint_map.items():
            a = landmarks[mcp_idx]
            b = landmarks[pip_idx]
            c = landmarks[dip_idx]
            angles[finger_name] = round(calculate_angle(a, b, c), 1)

        return angles

    def analyze(self, hand_landmarks, hand_id=0) -> dict:
        """
        Computes finger angles, tracks history, and determines movement direction
        (EXTENDING, BENDING, STATIONARY).
        """
        current_angles = self._get_finger_angles(hand_landmarks)
        movements = {}

        for finger_name, current_angle in current_angles.items():
            key = (hand_id, finger_name)

            if key not in self.angle_history:
                self.angle_history[key] = deque(maxlen=self.history_length)

            self.angle_history[key].append(current_angle)
            history = self.angle_history[key]

            if len(history) < self.history_length:
                motion = "STATIONARY"
            else:
                delta = history[-1] - history[0]
                if delta > 5.0:
                    motion = "EXTENDING"
                elif delta < -5.0:
                    motion = "BENDING"
                else:
                    motion = "STATIONARY"

            movements[finger_name] = {
                "angle": current_angle,
                "motion": motion
            }

        return movements


class BreathingAnalyzer:
    def __init__(self, raw_buffer_size=60, smooth_window_size=5):
        self.raw_buffer_size = raw_buffer_size
        self.smooth_window_size = smooth_window_size

        self.raw_signal = deque(maxlen=raw_buffer_size)
        self.smoothed_signal = deque(maxlen=raw_buffer_size)

    def update(self, left_shoulder, right_shoulder):
        """
        Calculates torso height signal, applies moving average smoothing,
        and stores signal history.
        """
        # Invert Y coordinate so upward torso movement increases value
        mid_shoulder_y = -((left_shoulder.y + right_shoulder.y) / 2.0)
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        norm_height = mid_shoulder_y / shoulder_width if shoulder_width > 0 else mid_shoulder_y
        self.raw_signal.append(norm_height)

        if len(self.raw_signal) >= self.smooth_window_size:
            recent_raw = list(self.raw_signal)[-self.smooth_window_size:]
            smoothed_val = sum(recent_raw) / self.smooth_window_size
            self.smoothed_signal.append(smoothed_val)

    def get_breathing_state(self) -> str:
        """
        Determines breathing direction based on smoothed signal trend.
        Returns: "IN", "OUT", or "NEUTRAL"
        """
        if len(self.smoothed_signal) < 10:
            return "NEUTRAL"

        delta = self.smoothed_signal[-1] - self.smoothed_signal[-10]

        if delta > 0.005:
            return "IN"
        elif delta < -0.005:
            return "OUT"
        else:
            return "NEUTRAL"

    def reset(self):
        self.raw_signal.clear()
        self.smoothed_signal.clear()