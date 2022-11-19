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
        self.handSense = .05
        
    def poseAnalyze(self, handPose):
        self.pose = handPose.landmark
        wrist = Point(self.pose[mp_hands.HandLandmark.WRIST])
        thumb_tip = Point(self.pose[mp_hands.HandLandmark.THUMB_TIP])
        
        index_mcp = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_MCP])
        index_dip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_DIP])
        index_tip = Point(self.pose[mp_hands.HandLandmark.INDEX_FINGER_TIP])
        
        middle_mcp = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
        middle_dip = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_DIP])
        middle_tip = Point(self.pose[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
        
        ring_mcp = Point(self.pose[mp_hands.HandLandmark.RING_FINGER_MCP])
        ring_dip = Point(self.pose[mp_hands.HandLandmark.RING_FINGER_DIP])
        ring_tip = Point(self.pose[mp_hands.HandLandmark.RING_FINGER_TIP])

        pinky_mcp = Point(self.pose[mp_hands.HandLandmark.PINKY_MCP])
        pinky_dip = Point(self.pose[mp_hands.HandLandmark.PINKY_DIP])
        pinky_tip = Point(self.pose[mp_hands.HandLandmark.PINKY_TIP])
        
        index_pnt = Point.distance(index_mcp, index_tip) + Point.distance(index_mcp, index_dip)
        middle_pnt = Point.distance(middle_mcp, middle_tip) + Point.distance(middle_mcp, middle_dip)
        ring_pnt = Point.distance(ring_mcp, ring_tip) + Point.distance(ring_mcp, ring_dip)
        pinky_pnt = Point.distance(pinky_mcp, pinky_tip) + Point.distance(pinky_mcp, pinky_dip)
        
        self.gesture = 0
        isIndexPoint = index_pnt > self.handSense
        isMiddlePoint = middle_pnt > self.handSense
        isRingPoint = ring_pnt > self.handSense
        isPinkyPoint = pinky_pnt > self.handSense
        
        #print(f"{isIndexPoint = }, {index_pnt:.3f}: {isMiddlePoint = }, {middle_pnt:.3f}: {isRingPoint = }, {ring_pnt:.3f}: {isPinkyPoint = }, {pinky_pnt:.3f}")
        
        if isIndexPoint:
            self.gesture = 1
        if isIndexPoint and isMiddlePoint:
            self.gesture = 2
        if isIndexPoint and isMiddlePoint and isRingPoint:
            self.gesture = 3
        if isIndexPoint and isMiddlePoint and isRingPoint and isPinkyPoint:
            self.gesture = 4