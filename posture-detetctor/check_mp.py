import mediapipe as mp
import cv2
print('mediapipe', mp.__version__)
print('cv2', cv2.__version__)
print('solutions', hasattr(mp, 'solutions'))
print('pose', hasattr(mp.solutions, 'pose'))
print('face_mesh', hasattr(mp.solutions, 'face_mesh'))
