import cv2
import mediapipe as mp
import math
import time
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# Distortion above this value produces a warning
DISTORTION_THRESHOLD = 60

# Distortion must remain above threshold for this many seconds
WARNING_TIME = 5

# Number of previous scores used for smoothing
SMOOTHING_FRAMES = 10

# Maximum deviation used for normalization
# This is an engineering threshold, NOT a medical threshold.
MAX_SHOULDER_DEVIATION = 20
MAX_HIP_DEVIATION = 20
MAX_BODY_AXIS_DEVIATION = 30


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
    Calculate midpoint between two MediaPipe landmarks.
    """

    return {
        "x": (p1.x + p2.x) / 2,
        "y": (p1.y + p2.y) / 2
    }


def distance(p1, p2):
    """
    Euclidean distance between two MediaPipe landmarks.
    """

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def point_distance(p1, p2):
    """
    Euclidean distance between two dictionaries
    containing x and y.
    """

    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 +
        (p1["y"] - p2["y"]) ** 2
    )


def line_angle(p1, p2):
    """
    Angle of a line in degrees.
    """

    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    return angle


def normalize_deviation(
    deviation,
    maximum
):
    """
    Convert deviation into a 0-100 score.
    """

    ratio = deviation / maximum

    ratio = max(
        0,
        min(ratio, 1)
    )

    return ratio * 100


# ============================================================
# POSTURE ANALYSIS
# ============================================================

def analyze_posture(landmarks):

    # --------------------------------------------------------
    # GET IMPORTANT LANDMARKS
    # --------------------------------------------------------

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

    left_ankle = landmarks[
        mp_pose.PoseLandmark.LEFT_ANKLE
    ]

    right_ankle = landmarks[
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]


    # --------------------------------------------------------
    # CHECK VISIBILITY
    # --------------------------------------------------------

    important_points = [
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip
    ]

    for point in important_points:

        if point.visibility < 0.5:
            return None


    # --------------------------------------------------------
    # CALCULATE MIDPOINTS
    # --------------------------------------------------------

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )


    # --------------------------------------------------------
    # 1. SHOULDER TILT
    # --------------------------------------------------------

    shoulder_angle = abs(
        line_angle(
            {
                "x": left_shoulder.x,
                "y": left_shoulder.y
            },
            {
                "x": right_shoulder.x,
                "y": right_shoulder.y
            }
        )
    )

    # Because a horizontal line can produce 180 degrees
    if shoulder_angle > 90:
        shoulder_angle = 180 - shoulder_angle


    shoulder_score = normalize_deviation(
        shoulder_angle,
        MAX_SHOULDER_DEVIATION
    )


    # --------------------------------------------------------
    # 2. HIP TILT
    # --------------------------------------------------------

    hip_angle = abs(
        line_angle(
            {
                "x": left_hip.x,
                "y": left_hip.y
            },
            {
                "x": right_hip.x,
                "y": right_hip.y
            }
        )
    )

    if hip_angle > 90:
        hip_angle = 180 - hip_angle


    hip_score = normalize_deviation(
        hip_angle,
        MAX_HIP_DEVIATION
    )


    # --------------------------------------------------------
    # 3. BODY AXIS
    # --------------------------------------------------------

    body_angle = line_angle(
        shoulder_mid,
        hip_mid
    )

    # We want deviation from vertical.
    #
    # In image coordinates, vertical is approximately 90 degrees.
    #
    # Convert angle to the closest distance from vertical.

    body_axis_deviation = abs(
        abs(body_angle) - 90
    )

    if body_axis_deviation > 90:
        body_axis_deviation = 180 - body_axis_deviation


    body_axis_score = normalize_deviation(
        body_axis_deviation,
        MAX_BODY_AXIS_DEVIATION
    )


    # --------------------------------------------------------
    # 4. TORSO LENGTH
    # --------------------------------------------------------

    torso_length = point_distance(
        shoulder_mid,
        hip_mid
    )


    # --------------------------------------------------------
    # 5. LEG LENGTH
    # --------------------------------------------------------

    left_leg_length = distance(
        left_hip,
        left_ankle
    )

    right_leg_length = distance(
        right_hip,
        right_ankle
    )

    average_leg_length = (
        left_leg_length +
        right_leg_length
    ) / 2


    # --------------------------------------------------------
    # 6. LEG / TORSO RATIO
    # --------------------------------------------------------

    if torso_length > 0:

        leg_torso_ratio = (
            average_leg_length /
            torso_length
        )

    else:

        leg_torso_ratio = 0


    # --------------------------------------------------------
    # PROPORTION SCORE
    # --------------------------------------------------------
    #
    # For now we don't classify this as "abnormal".
    # We simply display the ratio.
    #
    # A proper implementation should establish a
    # user-specific reference ratio.
    # --------------------------------------------------------

    proportion_score = 0


    # --------------------------------------------------------
    # FINAL DISTORTION SCORE
    # --------------------------------------------------------

    distortion_score = (

        shoulder_score * 0.20 +

        hip_score * 0.20 +

        body_axis_score * 0.60 +

        proportion_score * 0.00
    )


    distortion_score = min(
        distortion_score,
        100
    )


    # --------------------------------------------------------
    # RETURN EVERYTHING
    # --------------------------------------------------------

    return {

        "distortion": distortion_score,

        "shoulder_deviation":
            shoulder_angle,

        "hip_deviation":
            hip_angle,

        "body_axis_deviation":
            body_axis_deviation,

        "leg_torso_ratio":
            leg_torso_ratio,

        "shoulder_score":
            shoulder_score,

        "hip_score":
            hip_score,

        "body_axis_score":
            body_axis_score
    }


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")

    exit()


# ============================================================
# VARIABLES
# ============================================================

score_history = []

warning_start_time = None

posture_warning = False

paused = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print("Could not read camera frame.")

        break


    # --------------------------------------------------------
    # MIRROR IMAGE
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = pose.process(
        rgb_frame
    )


    # --------------------------------------------------------
    # DRAW BODY
    # --------------------------------------------------------

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        # ----------------------------------------------------
        # POSTURE ANALYSIS
        # ----------------------------------------------------

        if not paused:

            data = analyze_posture(
                results.pose_landmarks.landmark
            )


            if data is not None:

                current_score = data[
                    "distortion"
                ]


                # --------------------------------------------
                # SMOOTH SCORE
                # --------------------------------------------

                score_history.append(
                    current_score
                )

                if len(score_history) > SMOOTHING_FRAMES:

                    score_history.pop(0)


                smooth_score = np.mean(
                    score_history
                )


                # --------------------------------------------
                # WARNING LOGIC
                # --------------------------------------------

                if smooth_score >= DISTORTION_THRESHOLD:

                    if warning_start_time is None:

                        warning_start_time = time.time()


                    elapsed = (
                        time.time()
                        - warning_start_time
                    )


                    if elapsed >= WARNING_TIME:

                        posture_warning = True


                else:

                    warning_start_time = None

                    posture_warning = False


                # --------------------------------------------
                # DISPLAY SCORE
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"Distortion: {smooth_score:.1f}%",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )


                # --------------------------------------------
                # DISPLAY DETAILS
                # --------------------------------------------

                cv2.putText(
                    frame,
                    f"Shoulder: {data['shoulder_deviation']:.1f} deg",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Hip: {data['hip_deviation']:.1f} deg",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Body axis: {data['body_axis_deviation']:.1f} deg",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Leg/Torso: {data['leg_torso_ratio']:.2f}",
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )


                # --------------------------------------------
                # WARNING
                # --------------------------------------------

                if posture_warning:

                    cv2.putText(
                        frame,
                        "POSTURE WARNING",
                        (20, 215),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "Check your configured reference posture",
                        (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )


        else:

            # ------------------------------------------------
            # PAUSED
            # ------------------------------------------------

            cv2.putText(
                frame,
                "POSTURE MONITORING PAUSED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


    else:

        cv2.putText(
            frame,
            "BODY NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "P = Pause/Resume",
        (20, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "Posture Monitoring",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("p"):

        paused = not paused

        # Reset warning state when pausing
        warning_start_time = None
        posture_warning = False

        score_history.clear()


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pose.close()
