from operator import index
from geometry import Point
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
        
    def poseAnalyze(self, handPose):
        self.pose = handPose.landmark
        wrist = Point(self.pose[mp_hands.HandLandmark.WRIST])
        index_mcp = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_MCP])
        index_dip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_DIP])
        index_tip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_TIP])
        
        middle_mcp = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
        middle_tip = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
        
        pinky_mcp = Point(self.pose[mp_hands.HandLandmark.PINKY_MCP])
        pinky_tip = Point(self.pose[mp_hands.HandLandmark.PINKY_TIP])
        
        index_pnt = Point.distance(index_mcp, index_tip) + Point.distance(index_mcp, index_dip)
        
        self.gesture = 0
        isIndexPoint = index_pnt > 0.2
        if isIndexPoint:
            self.gesture = 1