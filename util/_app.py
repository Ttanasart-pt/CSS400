from tracemalloc import start
import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam       
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk 
import cv2
import threading
import time

from geometry import Point
from hand import Hand
import cvpainter
import brush

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class camApp(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self)
        
        self.debug = True
        self.root = parent
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        
        self.columnconfigure(0, minsize = 400)
        self.columnconfigure(1, minsize = 200)
        self.camFrame = tk.Frame(self)
        self.camFrame.grid(row=0, column=0)
        self.settingFrame = tk.Frame(self)
        self.settingFrame.grid(row=0, column=0)
        
        self.initSetting()
        self.initCamera()
        self.initCanvas()
        
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target = self.videoLoop, args = ())
        self.thread.start()
        
        self.detector = mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
        
        self.leftHand = Hand()
        self.rightHand = Hand()
        
        self.lastFrameTime = time.time_ns()
        self.frameTime = 0
        self.analyzeTime = 0
    
    def initSetting(self):
        self.settingFrame.grid(row = 0, column = 1, padx = 10, pady = 10, sticky="nsew")
        for i in range(5):
            self.settingFrame.rowconfigure(index = i, weight = 1)
        
        self.hsFrame = tk.Frame(self.settingFrame)
        self.hsFrame.columnconfigure(index = 0, minsize=200)
        self.hsFrame.columnconfigure(index = 1, weight = 1)
        self.hsFrame.pack(fill = 'both', expand = True)
        
        hsLabel = tk.Label(self.hsFrame, text = "Hand sensitivity")
        hsLabel.grid(row = 0, column = 0)
        
        self.handSenseValue = tk.DoubleVar(value = 0.2)
        self.handSense = tk.Spinbox(self.hsFrame, from_ = 0, to = 1, increment = 0.01,\
            textvariable = self.handSenseValue, command = self.onSensitivityChanged)
        self.handSense.grid(row = 0, column = 1, padx = 0, pady=10, sticky = "ew")
    
    def initCamera(self):
        self.camWidth = 800
        self.camHeight = 480
        
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.camWidth)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
        self.camView = None
        
    def initCanvas(self):
        self.laserPointerSurface = None
        self.laserThickness = 5
        self.laserColor = (255., 0., 0.)
        self.drawLastPos = None
        
        self.canvasSurface = None
        self.strokeDrawer = brush.Stroke(self.canvasSurface)
        
    def onSensitivityChanged(self):
        self.rightHand.handSense = self.handSenseValue.get()
        self.leftHand.handSense = self.handSenseValue.get()
    
    def start(self):
        self.root.mainloop()
    
    def surfaceCheck(self, result):
        if self.laserPointerSurface is None:
            self.laserPointerSurface = np.zeros_like(result)
        if self.canvasSurface is None:
            self.canvasSurface = np.zeros_like(result)
        
        self.strokeDrawer.setSurface(self.canvasSurface)
    
    def frameAnalyze(self, img):
        results = self.detector.process(img)
        self.surfaceCheck(img)
        
        if(results.multi_hand_landmarks):
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = mp_drawing_styles.get_default_hand_landmarks_style()
                )
                
                self.leftHand.poseAnalyze(hand_landmarks)
                
                if(self.leftHand.gesture == 1):
                    index_finger = self.leftHand.pose[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    fingerPos = Point(fromTuple = (round(index_finger.x * (img.shape[1])), round(index_finger.y * (img.shape[0])), 0))
                    if self.drawLastPos:
                        cvpainter.draw_line(self.laserPointerSurface, self.drawLastPos, fingerPos, self.laserThickness, self.laserColor)
                    self.drawLastPos = fingerPos
                    self.strokeDrawer.record(fingerPos)
                else:
                    self.drawLastPos = None
                    self.strokeDrawer.release()
        
        self.laserPointerSurface = np.clip(self.laserPointerSurface * 0.9, 0, None).astype(np.uint8)
        
        imgB = cv2.addWeighted(img, 1, self.laserPointerSurface, 1, 0.0)
        imgB = cv2.addWeighted(imgB, 1, self.canvasSurface, 1, 0.0)
        return imgB
    
    def debugInfo(self, img):
        fps = 1_000_000_000 / self.frameTime if self.frameTime != 0 else 0
        y = 32
        cv2.putText(img, f"fps: {fps:.2f}", (8, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 32
        cv2.putText(img, f"frame time: {self.frameTime / 1_000_000:.2f} ms", (8, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 32
        cv2.putText(img, f"analyze time: {self.analyzeTime / 1_000_000:.2f} ms", (8, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += 32
    
    def videoLoop(self):
        with pyvirtualcam.Camera(width = 1280, height = 720, fps = 30) as camOut:
            while True:
                startTime = time.time_ns()
                
                if self.stopEvent.is_set():
                    break
                
                stat, img = self.cam.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img = self.frameAnalyze(img)
                now = time.time_ns()
                self.analyzeTime = now - startTime
                
                if self.debug:
                    self.debugInfo(img)
                
                #camOut.send(img)
                #camOut.sleep_until_next_frame()
                
                image = Image.fromarray(img)
                image = ImageTk.PhotoImage(image)
                
                if self.camView:
                    self.camView.config(image = image)
                    self.camView.image = image
                else:
                    self.camView = tk.Label(self.camFrame, image = image)
                    self.camView.image = image
                    self.camView.grid(row = 0, column = 0, padx = 10, pady = 10, sticky="nsew")
                
                now = time.time_ns()
                self.frameTime = now - startTime
    
    def onClose(self):
        print("Closing...")  
        self.stopEvent.set()
        self.root.quit()
        
if __name__ == "__main__":
    print("Opening application...")
    time.sleep(1)
    
    root = tk.Tk()
    root.title("Live Cam")
    root.geometry("1200x500")
    
    app = camApp(root)
    app.pack(fill="both", expand=True)
    app.start()
        
   