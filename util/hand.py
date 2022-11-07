from util.geometry import Point
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def pointSide(a, b, c):
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) > 0;

class Hand():
    def __init__(self) -> None:
        self.gesture = 0
        self.pose = None
        self.pinchDistance = .05
        
    def poseAnalyze(self, handPose):
        self.pose = handPose.landmark
        wrist = Point(self.pose[mp_hands.HandLandmark.WRIST])
        thumb_tip = Point(self.pose[mp_hands.HandLandmark.THUMB_TIP])
        
        index_mcp = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_MCP])
        index_dip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_DIP])
        index_tip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_TIP])
        
        middle_mcp = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
        middle_tip = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
        
        pinky_mcp = Point(self.pose[mp_hands.HandLandmark.PINKY_MCP])
        pinky_tip = Point(self.pose[mp_hands.HandLandmark.PINKY_TIP])
        
        index_pnt = Point.distance(index_mcp, index_tip) + Point.distance(index_mcp, index_dip)
        isPinch = Point.distance(thumb_tip, index_tip) < self.pinchDistance
        
        self.gesture = 0
        if isPinch:
            self.gesture = 1