import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
detector = mp_holistic.Holistic(min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5)
    
def detectFace(img, results):
    mp_drawing.draw_landmarks(
        img,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()
    )

def detectPose(img, results):
    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style()
    )

def detectHandLeft(img, results):
    mp_drawing.draw_landmarks(
        img,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style()
    )
    
def detectHandRight(img, results):
    mp_drawing.draw_landmarks(
        img,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style()
    )
    
def main():
    camIn = cv2.VideoCapture(0)
    camIn.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camIn.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        _, img = camIn.read()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(img)

        detectPose(img, results)
        detectFace(img, results)
        detectHandLeft(img, results)
        detectHandRight(img, results)
        
        
if __name__ == "__main__":
    main()