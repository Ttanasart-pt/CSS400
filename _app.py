from tracemalloc import start
import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam       
import tkinter as tk
from tkinter import ttk
import sv_ttk

import keyboard
import copy
from PIL import Image, ImageTk 
import cv2
import threading
import time

from util.geometry import Point
from util._hand import Hand
import util.cvpainter as cvpainter
import util.brush as brush

import util.hough as hough

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
        
        self.onHandSettingChanged()
        
    def initSetting(self):        
        self.settingFrame.grid(row = 0, column = 1, padx = (0, 10), pady = 10, sticky="nsew")
        for i in range(2):
            self.settingFrame.rowconfigure(index = i, weight = 1)
        
        ##============================##
        self.hsFrame = ttk.LabelFrame(self.settingFrame, text = "Hand settings")
        self.hsFrame.columnconfigure(index = 0, minsize = 220)
        self.hsFrame.columnconfigure(index = 1, weight = 1)
        self.hsFrame.pack(fill = 'both', expand = True, padx = 20, pady = 20)
        
        ##============================##
        hsLabel = ttk.Label(self.hsFrame, text = "Hand sensitivity", justify = 'left')
        hsLabel.grid(row = 0, column = 0, sticky = 'W', padx = 20)
        
        self.handSenseValue = tk.DoubleVar(value = .3)
        self.handSense = ttk.Spinbox(self.hsFrame, from_ = 0, to = 1, increment = 0.01,\
            textvariable = self.handSenseValue, command = self.onHandSettingChanged)
        self.handSense.grid(row = 0, column = 1, padx = 10, pady=10, sticky = "ew")
    
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
        self.history = []
        self.points = []
        self.drawnPoints = None
        self.drawnShape = None
        self.lastGesture = 0
        self.currentTool = "None"
        self.canvasSurface = None
        self.strokeDrawer = brush.StrokeCapture(self.canvasSurface)
        
    def onHandSettingChanged(self):
        self.rightHand.handSense = self.handSenseValue.get()
        self.leftHand.handSense = self.handSenseValue.get()
    
    def start(self):
        self.root.mainloop()
    
    def surfaceCheck(self, result):
        if self.laserPointerSurface is None:
            self.laserPointerSurface = np.zeros_like(result)
        if self.canvasSurface is None:
            self.canvasSurface = np.zeros_like(result)
        if self.drawnPoints is None:
            self.drawnPoints = np.zeros_like(result)
        if self.drawnShape is None:
            self.drawnShape = np.zeros_like(result)
        if len(self.history) == 0:
            self.history.append(np.zeros_like(result))
        if len(self.history) == 1:
            self.history[0]=np.zeros_like(self.canvasSurface)

    def frameAnalyze(self, img):
        img = cv2.flip(img, 1)
        results = self.detector.process(img)
        self.surfaceCheck(img)
        
        drawing = False
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
                    self.points.append(fingerPos)
                    #self.strokeDrawer.record(fingerPos)
                    if(fingerPos.x < self.drawnPoints.shape[1] and fingerPos.y < self.drawnPoints.shape[0]):
                        self.drawnPoints[fingerPos.y][fingerPos.x] = 255
                        for i in range (0, 3):
                            for j in range (0, 3):
                                if(fingerPos.y+i < self.drawnPoints.shape[0] and fingerPos.x+j < self.drawnPoints.shape[1] and fingerPos.y-i > 0 and fingerPos.x-j > 0):
                                    self.drawnPoints[fingerPos.y + i][fingerPos.x-j] = 255
                                    self.drawnPoints[fingerPos.y + i][fingerPos.x+j] = 255
                                    self.drawnPoints[fingerPos.y -i][fingerPos.x-j] = 255
                                    self.drawnPoints[fingerPos.y -i][fingerPos.x+j] = 255
                elif(self.leftHand.gesture == 2):
                    self.lastGesture = self.leftHand.gesture
                    self.currentTool = "Line"
                elif(self.leftHand.gesture == 3):
                    self.lastGesture = self.leftHand.gesture
                    self.currentTool = "Shape"
                elif(self.leftHand.gesture == 4):
                    self.lastGesture = self.leftHand.gesture
                    self.currentTool = "Free"
                else:
                    if self.lastGesture == 2:
                        self.drawnShape = hough.detectLines(self.drawnPoints)
                    elif self.lastGesture == 3 and not np.all(self.drawnPoints==0):
                        self.drawnShape = hough.detectShape(self.drawnPoints)
                    elif self.lastGesture == 4 and not np.all(self.drawnPoints==0):
                        self.drawnShape = hough.drawLines(self.points, self.drawnShape)
                        self.points = []
                    if not np.all(self.drawnShape==0) or (self.lastGesture == 4 and not np.all(self.drawnPoints==0)):
                        self.canvasSurface = cv2.addWeighted(self.canvasSurface, 1, self.drawnShape, 1, 0.0)
                    self.drawLastPos = None
                    self.drawnPoints = np.zeros_like(self.canvasSurface)
                    self.drawnShape = np.zeros_like(self.canvasSurface)
                    self.Savehistory()
        
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
        cv2.putText(img, f"Current Tool: {self.currentTool}", (8, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
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

                if keyboard.is_pressed('delete'):
                    self.canvasSurface = np.zeros_like(img)
                    self.history.append(np.copy(self.canvasSurface))
                    self.drawnPoints = np.zeros_like(self.canvasSurface)
                    self.drawnShape = np.zeros_like(self.canvasSurface)
                
                if keyboard.is_pressed('up'):
                    self.undo()
                    self.drawnPoints = np.zeros_like(self.canvasSurface)
                    self.drawnShape = np.zeros_like(self.canvasSurface)
    
    def Savehistory(self):
        for i in range(0,len(self.history)):
            if np.array_equal(self.history[i], self.canvasSurface):
                return
        self.history.append(copy.deepcopy(self.canvasSurface))
        return 

    def undo(self):
        if len(self.history) == 1:
            return
        self.history.pop()
        self.canvasSurface = np.copy(self.history[len(self.history)-1])
        return

    def onClose(self):
        print("Closing...")  
        self.stopEvent.set()
        self.root.quit()
        
if __name__ == "__main__":
    print("Opening application...")
    time.sleep(1)
    
    root = tk.Tk()
    root.title("Live Cam")
    root.geometry("1320x500")
    
    sv_ttk.set_theme("dark")
    app = camApp(root)
    app.pack(fill="both", expand=True)
    app.start()