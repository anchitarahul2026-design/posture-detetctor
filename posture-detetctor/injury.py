import cv2
import mediapipe as mp
import math
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Number of frames used to smooth measurements
SMOOTHING_FRAMES = 15

# Threshold above which the posture is considered
# substantially different from the reference
WARNING_THRESHOLD = 60


# ============================================================
# MEDIAPIPE
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

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def midpoint(p1, p2):

    return {
        "x": (p1.x + p2.x) / 2,
        "y": (p1.y + p2.y) / 2
    }


def angle_between(p1, p2):

    dx = p2["x"] - p1["x"]
    dy = p2["y"] - p1["y"]

    return math.degrees(
        math.atan2(dy, dx)
    )


def normalize(value, maximum):

    score = value / maximum

    score = max(
        0,
        min(score, 1)
    )

    return score * 100


# ============================================================
# UPPER BODY ANALYSIS
# ============================================================

def analyze_upper_body(landmarks):

    LS = landmarks[
        mp_pose.PoseLandmark.LEFT_SHOULDER
    ]

    RS = landmarks[
        mp_pose.PoseLandmark.RIGHT_SHOULDER
    ]

    LE = landmarks[
        mp_pose.PoseLandmark.LEFT_ELBOW
    ]

    RE = landmarks[
        mp_pose.PoseLandmark.RIGHT_ELBOW
    ]

    LH = landmarks[
        mp_pose.PoseLandmark.LEFT_HIP
    ]

    RH = landmarks[
        mp_pose.PoseLandmark.RIGHT_HIP
    ]


    # --------------------------------------------------------
    # VISIBILITY
    # --------------------------------------------------------

    points = [
        LS, RS, LE, RE, LH, RH
    ]

    for point in points:

        if point.visibility < 0.5:

            return None


    # ========================================================
    # 1. SHOULDER HEIGHT DIFFERENCE
    # ========================================================

    shoulder_height_difference = abs(
        LS.y - RS.y
    )


    # ========================================================
    # 2. SHOULDER TILT
    # ========================================================

    shoulder_angle = abs(
        angle_between(
            {
                "x": LS.x,
                "y": LS.y
            },
            {
                "x": RS.x,
                "y": RS.y
            }
        )
    )

    if shoulder_angle > 90:

        shoulder_angle = (
            180 - shoulder_angle
        )


    # ========================================================
    # 3. SHOULDER WIDTH
    # ========================================================

    shoulder_width = distance(
        LS,
        RS
    )


    # ========================================================
    # 4. BODY CENTER
    # ========================================================

    shoulder_mid = midpoint(
        LS,
        RS
    )

    hip_mid = midpoint(
        LH,
        RH
    )


    # ========================================================
    # 5. UPPER BODY AXIS
    # ========================================================

    body_angle = angle_between(
        shoulder_mid,
        hip_mid
    )

    body_axis_deviation = abs(
        abs(body_angle) - 90
    )

    if body_axis_deviation > 90:

        body_axis_deviation = (
            180 - body_axis_deviation
        )


    # ========================================================
    # 6. LEFT/RIGHT ARM POSITION
    # ========================================================

    left_arm = distance(
        LS,
        LE
    )

    right_arm = distance(
        RS,
        RE
    )

    arm_asymmetry = abs(
        left_arm - right_arm
    )


    # ========================================================
    # NORMALIZED FEATURES
    # ========================================================

    shoulder_height_score = normalize(
        shoulder_height_difference,
        0.15
    )

    shoulder_tilt_score = normalize(
        shoulder_angle,
        20
    )

    body_axis_score = normalize(
        body_axis_deviation,
        25
    )

    arm_asymmetry_score = normalize(
        arm_asymmetry,
        0.15
    )


    # ========================================================
    # COMPOSITE SCORE
    # ========================================================

    distortion_score = (

        shoulder_height_score * 0.35 +

        shoulder_tilt_score * 0.25 +

        body_axis_score * 0.25 +

        arm_asymmetry_score * 0.15

    )


    return {

        "distortion":
            distortion_score,

        "shoulder_height":
            shoulder_height_difference,

        "shoulder_angle":
            shoulder_angle,

        "shoulder_width":
            shoulder_width,

        "body_axis":
            body_axis_deviation,

        "arm_asymmetry":
            arm_asymmetry,

        "shoulder_score":
            shoulder_height_score,

        "tilt_score":
            shoulder_tilt_score,

        "axis_score":
            body_axis_score,

        "arm_score":
            arm_asymmetry_score
    }


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open camera.")

    exit()


# ============================================================
# SCORE HISTORY
# ============================================================

score_history = []


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        break


    frame = cv2.flip(
        frame,
        1
    )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = pose.process(
        rgb
    )


    # ========================================================
    # BODY FOUND
    # ========================================================

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


        data = analyze_upper_body(
            results.pose_landmarks.landmark
        )


        if data is not None:

            # ------------------------------------------------
            # SMOOTH SCORE
            # ------------------------------------------------

            score_history.append(
                data["distortion"]
            )

            if len(score_history) > SMOOTHING_FRAMES:

                score_history.pop(0)


            average_score = np.mean(
                score_history
            )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Upper-body deviation: "
                f"{average_score:.1f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Shoulder asymmetry: "
                f"{data['shoulder_height']:.3f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Shoulder tilt: "
                f"{data['shoulder_angle']:.1f} deg",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Body axis: "
                f"{data['body_axis']:.1f} deg",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Arm asymmetry: "
                f"{data['arm_asymmetry']:.3f}",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # WARNING
            # ------------------------------------------------

            if average_score >= WARNING_THRESHOLD:

                cv2.putText(
                    frame,
                    "UNUSUAL UPPER-BODY POSTURE",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "Consider further assessment",
                    (20, 255),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
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
        "Q = Quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Upper Body Injury-Associated Posture Analysis",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pose.close()
