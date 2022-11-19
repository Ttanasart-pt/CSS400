import numpy as np
import cv2
import mediapipe as mp
import pyvirtualcam       
import tkinter as tk
from tkinter import ttk
import sv_ttk

from PIL import Image, ImageTk 
import cv2
import threading
import time

from util.geometry import Point
from util.hand import Hand
import util.cvpainter as painter
import util.brush as brush
from util.renderer import Renderer

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
        self.camFrame.grid(row = 0, column = 0)
        self.camFrame.rowconfigure(index = 0, weight = 10)
        self.camFrame.rowconfigure(index = 1, weight = 1)
        
        clear = ttk.Button(self.camFrame, text = "Clear", command = self.onClear)
        clear.grid(row = 1, column = 0, sticky = 'ew', padx = 20, pady = 20)
        
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
        
        self.result = None
        self.lastFrameTime = time.time_ns()
        self.frameTime = 0
        self.analyzeTime = 0
    
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
        
        self.handSenseValue = tk.DoubleVar(value = .05)
        self.handSense = ttk.Spinbox(self.hsFrame, from_ = 0, to = 1, increment = 0.01,\
            textvariable = self.handSenseValue, command = self.onHandSettingChanged)
        self.handSense.grid(row = 0, column = 1, padx = 10, pady=10, sticky = "ew")
        
        ##============================##
        self.bsFrame = ttk.LabelFrame(self.settingFrame, text = "Brush settings")
        self.bsFrame.columnconfigure(index = 0, minsize = 220)
        self.bsFrame.columnconfigure(index = 1, weight = 1)
        self.bsFrame.pack(fill = 'both', expand = True, padx = 20, pady = 20)
        
        ##============================##
        lsLabel = ttk.Label(self.bsFrame, text = "Line smoothing algorithm")
        lsLabel.grid(row = 0, column = 0, sticky = 'W', padx = 20)
        
        self.linesmValue = tk.StringVar()
        self.linesm = ttk.Combobox(self.bsFrame, textvariable = self.linesmValue, values = ('Distance accumulate', 'Moving average', 'Douglas-Peucker'), 
                                   state = "readonly")
        self.linesm.current(0)
        self.linesm.grid(row = 0, column = 1, padx = 10, pady=10, sticky = "ew")
        self.linesm.bind('<<ComboboxSelected>>', self.onBrushSettingChanged)
        
        ##============================##
        bsLabel = ttk.Label(self.bsFrame, text = "Brush size")
        bsLabel.grid(row = 1, column = 0, sticky = 'W', padx = 20)
        
        self.brushSizeValue = tk.IntVar(value = 4)
        self.brushSize = ttk.Spinbox(self.bsFrame, from_ = 0, to = 16, increment = 1,\
            textvariable = self.brushSizeValue, command = self.onBrushSettingChanged)
        self.brushSize.grid(row = 1, column = 1, padx = 10, pady=10, sticky = "ew")
        
        ##============================##
        bsLabel = ttk.Label(self.bsFrame, text = "Pointer size")
        bsLabel.grid(row = 2, column = 0, sticky = 'W', padx = 20)
        
        self.pointerSizeValue = tk.IntVar(value = 4)
        self.pointerSize = ttk.Spinbox(self.bsFrame, from_ = 0, to = 16, increment = 1,\
            textvariable = self.pointerSizeValue, command = self.onBrushSettingChanged)
        self.pointerSize.grid(row = 2, column = 1, padx = 10, pady=10, sticky = "ew")
        
        ##============================##
        bsLabel = ttk.Label(self.bsFrame, text = "Line distance")
        bsLabel.grid(row = 3, column = 0, sticky = 'W', padx = 20)
        
        self.lineDistValue = tk.IntVar(value = 8)
        self.lineDist = ttk.Spinbox(self.bsFrame, from_ = 0, to = 32, increment = 1,\
            textvariable = self.lineDistValue, command = self.onBrushSettingChanged)
        self.lineDist.grid(row = 3, column = 1, padx = 10, pady=10, sticky = "ew")
        
        ##============================##
        bsLabel = ttk.Label(self.bsFrame, text = "Average pool")
        bsLabel.grid(row = 4, column = 0, sticky = 'W', padx = 20)
        
        self.poolValue = tk.IntVar(value = 4)
        self.pool = ttk.Spinbox(self.bsFrame, from_ = 0, to = 16, increment = 1,\
            textvariable = self.poolValue, command = self.onBrushSettingChanged)
        self.pool.grid(row = 4, column = 1, padx = 10, pady=10, sticky = "ew")
        
        ##============================##
        bsLabel = ttk.Label(self.bsFrame, text = "Douglas threshold")
        bsLabel.grid(row = 5, column = 0, sticky = 'W', padx = 20)
        
        self.douglasValue = tk.IntVar(value = 4)
        self.douglas = ttk.Spinbox(self.bsFrame, from_ = 0, to = 32, increment = 1,\
            textvariable = self.douglasValue, command = self.onBrushSettingChanged)
        self.douglas.grid(row = 5, column = 1, padx = 10, pady=10, sticky = "ew")
        
    
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
        self.renderer = Renderer()
        self.strokeDrawer = brush.StrokeCapture(self.renderer)
        
    def onHandSettingChanged(self, _ = None):
        self.rightHand.handSense = self.handSenseValue.get()
        self.leftHand.handSense = self.handSenseValue.get()
    
    def onBrushSettingChanged(self, _ = None):
        self.laserThickness = self.pointerSizeValue.get()
        self.strokeDrawer.thickness = self.brushSizeValue.get()
        self.strokeDrawer.smoothAlgo = self.linesmValue.get()
        
        self.strokeDrawer.lineMinDistance = self.lineDistValue.get()
        self.strokeDrawer.nAverage = self.poolValue.get()
        self.strokeDrawer.douglasThres = self.douglasValue.get()
    
    def start(self):
        self.root.mainloop()
    
    def surfaceCheck(self, result):
        self.result = result
        if self.laserPointerSurface is None:
            self.laserPointerSurface = np.zeros_like(result)
        if self.canvasSurface is None:
            self.canvasSurface = np.zeros_like(result)
    
    def frameAnalyze(self, img):
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
                        painter.draw_line(self.laserPointerSurface, self.drawLastPos, fingerPos, self.laserThickness, self.laserColor)
                    self.drawLastPos = fingerPos
                    drawing = True
                    self.strokeDrawer.record(fingerPos)
                else:
                    if not drawing and self.strokeDrawer.isDrawing:
                        self.strokeDrawer.release()
                    self.drawLastPos = None
        
        self.laserPointerSurface = np.clip(self.laserPointerSurface * 0.9, 0, None).astype(np.uint8)
        self.canvasSurface = self.renderer.render(self.canvasSurface)
        
        cont = cv2.cvtColor(self.canvasSurface, cv2.COLOR_RGB2GRAY)
        _, cont = cv2.threshold(cont, 1, 255, cv2.THRESH_BINARY)
        cont = cv2.cvtColor(cont, cv2.COLOR_GRAY2RGB)
        cont = cont.astype(float) / 255
        canvasMasked = self.canvasSurface.astype(float) * cont
        imageMasked  = img.astype(float) * (1 - cont)
        imageCanvas = (canvasMasked + imageMasked).astype(np.uint8)
        
        #imgB = cv2.addWeighted(img, 1, self.canvasSurface, 1, 0.0)
        imgB = cv2.addWeighted(imageCanvas, 1, self.laserPointerSurface, 1, 0.0)
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
                
                _, img = self.cam.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.flip(img, 1)
                
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
    
    def onClear(self):
        self.renderer.clear()
    
    def onClose(self):
        print("Closing...")  
        self.stopEvent.set()
        self.root.quit()
        
if __name__ == "__main__":
    print("Opening application...")
    time.sleep(1)
    
    root = tk.Tk()
    root.title("LiveCanvas")
    root.geometry("1320x580")
    
    sv_ttk.set_theme("dark")
    app = camApp(root)
    app.pack(fill="both", expand=True)
    app.start()
        
   