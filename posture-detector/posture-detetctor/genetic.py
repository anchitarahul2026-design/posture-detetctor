import math


def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2
    )


def angle_between(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    angle = math.degrees(math.atan2(dy, dx))

    return angle


def midpoint(p1, p2):
    return {
        "x": (p1.x + p2.x) / 2,
        "y": (p1.y + p2.y) / 2
    }


def point_distance(p1, p2):
    return math.sqrt(
        (p1["x"] - p2["x"])**2 +
        (p1["y"] - p2["y"])**2
    )


def posture_distortion(landmarks):

    # -------------------------
    # GET LANDMARKS
    # -------------------------

    LS = landmarks["left_shoulder"]
    RS = landmarks["right_shoulder"]

    LH = landmarks["left_hip"]
    RH = landmarks["right_hip"]

    LA = landmarks["left_ankle"]
    RA = landmarks["right_ankle"]

    # -------------------------
    # MIDPOINTS
    # -------------------------

    shoulder_mid = midpoint(LS, RS)
    hip_mid = midpoint(LH, RH)

    # -------------------------
    # SHOULDER TILT
    # -------------------------

    shoulder_angle = angle_between(LS, RS)

    # reference = horizontal
    shoulder_deviation = abs(shoulder_angle)

    # -------------------------
    # HIP TILT
    # -------------------------

    hip_angle = angle_between(LH, RH)

    hip_deviation = abs(hip_angle)

    # -------------------------
    # TORSO LENGTH
    # -------------------------

    torso_length = point_distance(
        shoulder_mid,
        hip_mid
    )

    # -------------------------
    # LEG LENGTH
    # -------------------------

    left_leg = distance(LH, LA)
    right_leg = distance(RH, RA)

    average_leg = (left_leg + right_leg) / 2

    # -------------------------
    # LEG / TORSO RATIO
    # -------------------------

    if torso_length != 0:
        leg_torso_ratio = average_leg / torso_length
    else:
        leg_torso_ratio = 0

    # -------------------------
    # SCORE
    # -------------------------

    # Example reference values
    NORMAL_SHOULDER = 0
    NORMAL_HIP = 0
    NORMAL_LEG_TORSO = 1.5

    shoulder_error = abs(
        shoulder_deviation - NORMAL_SHOULDER
    )

    hip_error = abs(
        hip_deviation - NORMAL_HIP
    )

    ratio_error = abs(
        leg_torso_ratio - NORMAL_LEG_TORSO
    )

    # Weighted score
    distortion_score = (
        shoulder_error * 0.30 +
        hip_error * 0.30 +
        ratio_error * 20 * 0.40
    )

    return {
        "shoulder_deviation": shoulder_deviation,
        "hip_deviation": hip_deviation,
        "leg_torso_ratio": leg_torso_ratio,
        "distortion_score": distortion_score
    }
