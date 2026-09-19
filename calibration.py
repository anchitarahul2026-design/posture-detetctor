# calibration.py - Personal Full-Body Calibration System

import time
import statistics
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import json
import os

from geometry import (
    extract_all_body_measurements,
    is_landmark_valid,
    distance,
    midpoint,
)


class CalibrationState:
    """State machine for the calibration process."""
    
    IDLE = "idle"
    COUNTDOWN = "countdown"
    NEUTRAL = "neutral"
    HEAD = "head"
    SHOULDERS = "shoulders"
    BREATHING = "breathing"
    HANDS = "hands"
    COMPLETE = "complete"
    MONITORING = "monitoring"
    
    # Human-readable labels for each state
    LABELS = {
        IDLE: "Ready to calibrate",
        COUNTDOWN: "Get ready...",
        NEUTRAL: "Sit naturally and keep your normal posture",
        HEAD: "Slowly look left, return center, look right, return center",
        SHOULDERS: "Raise both shoulders slightly, then relax",
        BREATHING: "Breathe normally for a few seconds",
        HANDS: "Keep your hands naturally visible",
        COMPLETE: "Calibration complete! Starting monitor...",
        MONITORING: "Monitoring",
    }
    
    # Minimum sample counts for each stage
    MIN_SAMPLES = {
        NEUTRAL: 30,  # ~1 second at 30fps
        HEAD: 60,
        SHOULDERS: 30,
        BREATHING: 60,
        HANDS: 30,
    }
    
    def __init__(self):
        self.current_state = self.IDLE
        self.stage_samples = {}
        self.stage_start_time = None
        self.countdown_value = 3
        
    def start(self):
        """Start the calibration sequence."""
        self.current_state = self.COUNTDOWN
        self.countdown_value = 3
        self.stage_start_time = time.time()
        self.stage_samples = {}
        
    def advance(self):
        """Advance to the next calibration stage."""
        if self.current_state == self.COUNTDOWN:
            self.current_state = self.NEUTRAL
        elif self.current_state == self.NEUTRAL:
            self.current_state = self.HEAD
        elif self.current_state == self.HEAD:
            self.current_state = self.SHOULDERS
        elif self.current_state == self.SHOULDERS:
            self.current_state = self.BREATHING
        elif self.current_state == self.BREATHING:
            self.current_state = self.HANDS
        elif self.current_state == self.HANDS:
            self.current_state = self.COMPLETE
        elif self.current_state == self.COMPLETE:
            self.current_state = self.MONITORING
            
        self.stage_start_time = time.time()
        self.stage_samples[self.current_state] = []
        
    def get_label(self) -> str:
        """Get the human-readable label for the current state."""
        return self.LABELS.get(self.current_state, "Unknown")
    
    def get_progress(self) -> float:
        """Get progress through calibration (0.0 to 1.0)."""
        states = [self.NEUTRAL, self.HEAD, self.SHOULDERS, 
                  self.BREATHING, self.HANDS]
        if self.current_state == self.IDLE:
            return 0.0
        if self.current_state == self.COMPLETE or self.current_state == self.MONITORING:
            return 1.0
        if self.current_state == self.COUNTDOWN:
            return 0.05
        
        try:
            idx = states.index(self.current_state)
            return (idx + 1) / len(states)
        except ValueError:
            return 0.0


class CalibrationCollector:
    """
    Collects and aggregates body measurements during calibration.
    Uses robust statistics (median) to build a personal baseline.
    """
    
    def __init__(self, min_samples=10, confidence_threshold=0.5):
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.samples = {}
        self.baseline = None
        
    def add_sample(self, measurements: Dict[str, Any]) -> bool:
        """
        Add a sample to the collection.
        Returns True if sample was accepted, False otherwise.
        """
        if not measurements.get("valid", False):
            return False
        
        # Check confidence of key landmarks
        confidences = measurements.get("confidences", {})
        for lm, conf in confidences.items():
            if conf < self.confidence_threshold:
                return False
        
        # Store the sample
        for key, value in measurements.items():
            if key in ["valid", "confidences", "shoulder_center", "hip_center"]:
                continue
            if key not in self.samples:
                self.samples[key] = []
            self.samples[key].append(value)
        
        return True
    
    def compute_baseline(self) -> Dict[str, Any]:
        """
        Compute the baseline using median values.
        Returns a dictionary with baseline measurements.
        """
        baseline = {}
        
        for key, values in self.samples.items():
            if len(values) < self.min_samples:
                continue
            
            # Use median for robustness against outliers
            try:
                median_value = statistics.median(values)
                baseline[key] = median_value
                baseline[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
                baseline[f"{key}_n"] = len(values)
            except (statistics.StatisticsError, TypeError):
                continue
        
        baseline["_metadata"] = {
            "created_at": time.time(),
            "sample_count": len(self.samples.get("head_forward_ratio", [])),
            "version": 1
        }
        
        self.baseline = baseline
        return baseline
    
    def get_sample_count(self) -> int:
        """Get the number of samples collected."""
        if not self.samples:
            return 0
        # Return the count from the first key
        first_key = next(iter(self.samples.keys()))
        return len(self.samples[first_key])
    
    def reset(self):
        """Reset the collector."""
        self.samples = {}
        self.baseline = None


class PersonalBaseline:
    """
    Manages the personal baseline: storage, loading, saving, and comparison.
    """
    
    def __init__(self, data_dir="data/calibration"):
        self.data_dir = data_dir
        self.baseline = None
        self.current_deviations = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def set_baseline(self, baseline: Dict[str, Any]):
        """Set the baseline from calibration."""
        self.baseline = baseline
        
    def load(self, filename="baseline.json") -> bool:
        """Load a baseline from file."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                self.baseline = json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False
    
    def save(self, filename="baseline.json") -> bool:
        """Save the baseline to file."""
        if not self.baseline:
            return False
        
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(self.baseline, f, indent=2)
            return True
        except IOError:
            return False
    
    def compare(self, measurements: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare current measurements against the baseline.
        Returns deviation scores (0.0 = identical, 1.0 = very different).
        """
        if not self.baseline or not measurements.get("valid", False):
            return {}
        
        deviations = {}
        
        # List of measurements to compare
        compare_keys = [
            "head_forward_ratio", "head_tilt", "head_rotation",
            "shoulder_tilt", "shoulder_symmetry",
            "torso_angle", "torso_lateral_lean", "torso_forward_lean",
            "hip_tilt", "hip_symmetry",
            "left_knee_angle", "right_knee_angle", "leg_symmetry",
        ]
        
        for key in compare_keys:
            if key not in measurements or key not in self.baseline:
                continue
            
            current = measurements[key]
            baseline_val = self.baseline[key]
            
            # Get expected range (standard deviation from calibration)
            std_key = f"{key}_std"
            expected_range = self.baseline.get(std_key, 5.0)
            
            # Calculate normalized deviation
            if expected_range > 0:
                deviation = abs(current - baseline_val) / expected_range
                # Clamp to [0, 1]
                deviation = min(1.0, deviation)
            else:
                # If no variance, use a small tolerance
                tolerance = 1.0
                deviation = min(1.0, abs(current - baseline_val) / tolerance)
            
            deviations[key] = deviation
        
        self.current_deviations = deviations
        return deviations
    
    def get_overall_score(self, deviations: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate an overall body alignment score (0-100).
        Higher = better alignment with baseline.
        """
        if deviations is None:
            deviations = self.current_deviations
        
        if not deviations:
            return 100.0
        
        # Average deviation, convert to score (100 = perfect, 0 = worst)
        avg_deviation = sum(deviations.values()) / len(deviations)
        score = max(0.0, min(100.0, (1.0 - avg_deviation) * 100))
        return score
    
    def get_detailed_state(self, deviations: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get a detailed body state breakdown.
        """
        if deviations is None:
            deviations = self.current_deviations
        
        if not deviations or not self.baseline:
            return {"overall_score": 100.0, "components": {}}
        
        # Group deviations by body region
        groups = {
            "head": ["head_forward_ratio", "head_tilt", "head_rotation"],
            "shoulders": ["shoulder_tilt", "shoulder_symmetry"],
            "torso": ["torso_angle", "torso_lateral_lean", "torso_forward_lean"],
            "hips": ["hip_tilt", "hip_symmetry"],
            "legs": ["left_knee_angle", "right_knee_angle", "leg_symmetry"],
        }
        
        result = {
            "overall_score": self.get_overall_score(deviations),
            "components": {},
            "deviations": deviations,
        }
        
        for group_name, keys in groups.items():
            group_deviations = [d for k, d in deviations.items() if k in keys]
            if group_deviations:
                avg_dev = sum(group_deviations) / len(group_deviations)
                result["components"][group_name] = {
                    "score": max(0.0, (1.0 - avg_dev) * 100),
                    "deviation_avg": avg_dev,
                    "keys_used": keys,
                }
        
        return result


class CalibrationManager:
    """
    Main calibration manager that orchestrates the entire calibration process.
    """
    
    def __init__(self, confidence_threshold=0.5, min_samples=10):
        self.state = CalibrationState()
        self.collector = CalibrationCollector(min_samples, confidence_threshold)
        self.baseline = PersonalBaseline()
        self.confidence_threshold = confidence_threshold
        
        # For tracking head movement during HEAD stage
        self.head_positions = deque(maxlen=30)
        
    def update(self, landmarks, hand_landmarks=None) -> Dict[str, Any]:
        """
        Process a frame during calibration.
        Returns a status dictionary.
        """
        if self.state.current_state == CalibrationState.IDLE:
            return {"state": "idle", "is_calibrating": False}
        
        if self.state.current_state == CalibrationState.COUNTDOWN:
            return self._handle_countdown()
        
        if self.state.current_state == CalibrationState.COMPLETE:
            return {"state": "complete", "is_calibrating": False}
        
        if self.state.current_state == CalibrationState.MONITORING:
            return {"state": "monitoring", "is_calibrating": False}
        
        # For all other states (NEUTRAL, HEAD, SHOULDERS, BREATHING, HANDS)
        return {"state": self.state.current_state, "is_calibrating": True}
    
    def _handle_countdown(self) -> Dict[str, Any]:
        """Handle the countdown state."""
        return {
            "state": "countdown",
            "countdown": self.state.countdown_value,
            "is_calibrating": True
        }