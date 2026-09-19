import cv2
import mediapipe as mp
import math
import time
import winsound


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum downward movement between frames
VERTICAL_MOVEMENT_THRESHOLD = 0.08

# Body orientation threshold
# Larger value = more horizontal
HORIZONTAL_ANGLE_THRESHOLD = 60

# How long the suspicious position must remain
# before declaring a fall
FALL_CONFIRMATION_TIME = 1.5

# How long the alarm should sound
ALARM_DURATION = 5


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def midpoint(p1, p2):
    """
    Calculate midpoint of two landmarks.
    """

    return {
        "x": (p1.x + p2.x) / 2,
        "y": (p1.y + p2.y) / 2
    }


def calculate_angle(p1, p2):
    """
    Calculate angle of line between two points.
    """

    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    return angle


def get_body_information(landmarks):

    left_shoulder = landmarks[
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]

    right_shoulder = landmarks[
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]

    left_hip = landmarks[
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    right_hip = landmarks[
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    # --------------------------------------------------------
    # Check visibility
    # --------------------------------------------------------

    required_points = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]

    for point in required_points:

        if point.visibility < 0.5:

            return None


    # --------------------------------------------------------
    # Shoulder midpoint
    # --------------------------------------------------------

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )


    # --------------------------------------------------------
    # Hip midpoint
    # --------------------------------------------------------

    hip_mid = midpoint(
        left_hip,
        right_hip
    )


    # --------------------------------------------------------
    # Overall body center
    # --------------------------------------------------------

    body_center = {

        "x":
            (shoulder_mid["x"] +
             hip_mid["x"]) / 2,

        "y":
            (shoulder_mid["y"] +
             hip_mid["y"]) / 2
    }


    # --------------------------------------------------------
    # Body axis
    # --------------------------------------------------------

    angle = calculate_angle(
        shoulder_mid,
        hip_mid
    )


    # --------------------------------------------------------
    # Convert to deviation from vertical
    # --------------------------------------------------------

    vertical_deviation = abs(
        abs(angle) - 90
    )

    if vertical_deviation > 90:

        vertical_deviation = (
            180 - vertical_deviation
        )


    return {

        "body_center": body_center,

        "shoulder_mid":
            shoulder_mid,

        "hip_mid":
            hip_mid,

        "angle":
            angle,

        "vertical_deviation":
            vertical_deviation
    }


# ============================================================
# FALL DETECTOR CLASS
# ============================================================

class FallDetector:

    def __init__(self):

        self.previous_center = None

        self.fall_start_time = None

        self.fall_detected = False

        self.paused = False


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.previous_center = None

        self.fall_start_time = None

        self.fall_detected = False


    # --------------------------------------------------------
    # TOGGLE PAUSE
    # --------------------------------------------------------

    def toggle_pause(self):

        self.paused = not self.paused

        self.reset()


    # --------------------------------------------------------
    # ANALYZE FRAME
    # --------------------------------------------------------

    def analyze(self, body_data):

        if self.paused:

            return {

                "fall": False,

                "suspicious": False,

                "vertical_change": 0,

                "orientation": 0,

                "status": "PAUSED"
            }


        if body_data is None:

            return {

                "fall": False,

                "suspicious": False,

                "vertical_change": 0,

                "orientation": 0,

                "status": "BODY NOT DETECTED"
            }


        current_center = (
            body_data["body_center"]
        )


        # ----------------------------------------------------
        # FIRST FRAME
        # ----------------------------------------------------

        if self.previous_center is None:

            self.previous_center = (
                current_center
            )

            return {

                "fall": False,

                "suspicious": False,

                "vertical_change": 0,

                "orientation":
                    body_data["vertical_deviation"],

                "status": "MONITORING"
            }


        # ----------------------------------------------------
        # VERTICAL MOVEMENT
        # ----------------------------------------------------

        vertical_change = (
            current_center["y"]
            - self.previous_center["y"]
        )


        # ----------------------------------------------------
        # BODY ORIENTATION
        # ----------------------------------------------------

        orientation = (
            body_data["vertical_deviation"]
        )


        # ----------------------------------------------------
        # DETECT SUSPICIOUS EVENT
        # ----------------------------------------------------

        sudden_downward_motion = (
            vertical_change
            > VERTICAL_MOVEMENT_THRESHOLD
        )


        body_is_horizontal = (
            orientation
            > HORIZONTAL_ANGLE_THRESHOLD
        )


        suspicious = (
            sudden_downward_motion
            and body_is_horizontal
        )


        # ----------------------------------------------------
        # CONFIRMATION TIMER
        # ----------------------------------------------------

        if suspicious:

            if self.fall_start_time is None:

                self.fall_start_time = time.time()


            elapsed = (
                time.time()
                - self.fall_start_time
            )


            if elapsed >= FALL_CONFIRMATION_TIME:

                self.fall_detected = True


        else:

            # Only reset the timer if a fall hasn't
            # already been confirmed.

            if not self.fall_detected:

                self.fall_start_time = None


        # ----------------------------------------------------
        # UPDATE PREVIOUS POSITION
        # ----------------------------------------------------

        self.previous_center = (
            current_center
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if self.fall_detected:

            status = "FALL DETECTED"

        elif suspicious:

            status = "POSSIBLE FALL"

        else:

            status = "MONITORING"


        return {

            "fall":
                self.fall_detected,

            "suspicious":
                suspicious,

            "vertical_change":
                vertical_change,

            "orientation":
                orientation,

            "status":
                status
        }


# ============================================================
# ALARM
# ============================================================

def trigger_alarm():

    print()
    print("==============================")
    print("        FALL DETECTED")
    print("==============================")
    print("Please check the person.")
    print()


    # Windows system alarm
    #
    # Frequency = 2500 Hz
    # Duration = ALARM_DURATION seconds

    winsound.Beep(
        2500,
        ALARM_DURATION * 1000
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ============================================================
# FALL DETECTOR
# ============================================================

detector = FallDetector()


# ============================================================
# ALARM CONTROL
# ============================================================

alarm_triggered = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()


    if not success:

        print("Could not read camera.")

        break


    # --------------------------------------------------------
    # Mirror image
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    results = pose.process(
        rgb_frame
    )


    body_data = None


    # --------------------------------------------------------
    # Detect body
    # --------------------------------------------------------

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        body_data = get_body_information(
            results.pose_landmarks.landmark
        )


    # --------------------------------------------------------
    # Analyze fall
    # --------------------------------------------------------

    result = detector.analyze(
        body_data
    )


    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    status = result["status"]


    if status == "FALL DETECTED":

        cv2.putText(
            frame,
            "FALL DETECTED!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )


        cv2.putText(
            frame,
            "CHECK PERSON",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # Trigger alarm only once
        # ----------------------------------------------------

        if not alarm_triggered:

            alarm_triggered = True

            trigger_alarm()


    elif status == "POSSIBLE FALL":

        cv2.putText(
            frame,
            "POSSIBLE FALL",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            3
        )


    elif status == "PAUSED":

        cv2.putText(
            frame,
            "FALL MONITORING PAUSED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    elif status == "BODY NOT DETECTED":

        cv2.putText(
            frame,
            "BODY NOT DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "MONITORING",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Vertical movement: "
        f"{result['vertical_change']:.3f}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Body orientation: "
        f"{result['orientation']:.1f} deg",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "P = Pause / Resume",
        (20, frame.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "R = Reset alarm",
        (20, frame.shape[0] - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q = Quit",
        (20, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Fall Detection",
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == ord("q"):

        break


    # Pause
    elif key == ord("p"):

        detector.toggle_pause()

        alarm_triggered = False


    # Reset
    elif key == ord("r"):

        detector.reset()

        alarm_triggered = False


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pose.close()
