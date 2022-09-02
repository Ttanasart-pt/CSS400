import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam       
import tkinter as tk

from PIL import Image, ImageTk 
import cv2
import threading
import time

from geometry import Point
import cvpainter
import brush

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
        self.pose = handPose
        wrist = handPose.landmark[mp_hands.HandLandmark.WRIST]
        index_mcp = handPose.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = handPose.landmark[mp_hands.HandLandmark.PINKY_MCP]
        palm = Point((wrist.x + index_mcp.x + pinky_mcp.x) / 3, (wrist.y + index_mcp.y + pinky_mcp.y) / 3, (wrist.z + index_mcp.z + pinky_mcp.z) / 3)
        
        index_tip = handPose.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        self.gesture = 0
        
        isIndexPoint = pointSide(index_mcp, pinky_mcp, index_tip) != pointSide(index_mcp, pinky_mcp, wrist)
        if isIndexPoint:
            self.gesture = 1

class camApp:
    def __init__(self) -> None:
        self.camWidth = 800
        self.camHeight = 480
        
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.camWidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
        self.frame = None
        self.panel = None
          
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target = self.videoLoop, args = ())
        self.thread.start()
        
        self.root = tk.Tk()
        self.root.title("Live Cam")
        self.root.geometry("800x500")
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        
        self.detector = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
        
        self.leftHand = Hand()
        self.rightHand = Hand()
        
        self.laserPointerSurface = np.empty((self.camHeight, self.camWidth, 3), dtype = np.uint8)
        self.laserThickness = 5
        self.laserColor = (255., 0., 0.)
        self.drawLastPos = None
        
        self.canvasSurface = np.empty((self.camHeight, self.camWidth, 3), dtype = np.uint8)
        
        self.strokeDrawer = brush.Stroke(self.canvasSurface)

        tk.Button(self.root, text = "Button", command = self.onButton)\
          .pack(side = "bottom", fill = "both", expand = "yes", padx = 10, pady = 10)
        
    def start(self):
        self.root.mainloop()
        
    def onButton(self):
        pass
    
    def frameAnalyze(self, img):
        results = self.detector.process(img)
        
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles
            .get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles
            .get_default_hand_landmarks_style())
        mp_drawing.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles
            .get_default_hand_landmarks_style())
        
        if(results.right_hand_landmarks):
            self.rightHand.poseAnalyze(results.right_hand_landmarks)
        
        if(results.left_hand_landmarks):
            self.leftHand.poseAnalyze(results.left_hand_landmarks)
            
            if(self.leftHand.gesture == 1):
                index_finger = self.leftHand.pose[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingerPos = (round(index_finger.x * (img.shape[1])), round(index_finger.y * (img.shape[0])))
                if self.drawLastPos:
                    cvpainter.draw_line(self.laserPointerSurface, self.drawLastPos, fingerPos, self.laserColor, self.laserThickness)
                self.drawLastPos = fingerPos
                self.strokeDrawer.record(fingerPos)
            else:
                self.drawLastPos = None
                self.strokeDrawer.release()
        
        self.laserPointerSurface = np.clip(self.laserPointerSurface * 0.9, 0, None).astype(np.uint8)
        
        imgB = cv2.addWeighted(img, 1, self.laserPointerSurface, 1, 0.0)
        imgB = cv2.addWeighted(imgB, 1, self.canvasSurface, 1, 0.0)
        return imgB
    
    def videoLoop(self):
        with pyvirtualcam.Camera(width = 1280, height = 720, fps = 30) as camOut:
            while True:
                if self.stopEvent.is_set():
                    break
                
                _, img = self.cam.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.frameAnalyze(img)
                
                #camOut.send(img)
                #camOut.sleep_until_next_frame()
                
                image = Image.fromarray(img)
                image = ImageTk.PhotoImage(image)
                
                if self.panel:
                    self.panel.config(image = image)
                    self.panel.image = image
                else:
                    self.panel = tk.Label(image = image)
                    self.panel.image = image
                    self.panel.pack(side = "left", padx = 10, pady = 10)
    
    def onClose(self):
        print("Closing...")
        self.stopEvent.set()
        self.root.quit()

if __name__ == "__main__":
    print("Opening application...")
    time.sleep(1)
    
    app = camApp()
    app.start()
    