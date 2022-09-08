from geometry import Point
import cvpainter
import cv2

class Stroke():
    def __init__(self, surface) -> None:
        self.anchors = []
        self.surface = surface
        self.thickness = 5
        self.color = (0, 255, 255)
        
        self.anchorMinDistance = 32
        self.lineMinDistance = 32
    
    def setSurface(self, surface):
        self.surface = surface
    
    def record(self, point):
        self.anchors.append(point)
    
    def fitCircle(self):
        
        return False
    
    def drawPath(self):
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                continue
            if(Point.distance(op, p) < self.anchorMinDistance):
                continue
            
            cvpainter.draw_line(self.surface, op, p, self.thickness, self.color)
            op = p
    
    def release(self):
        if(len(self.anchors) == 0):
            return
        
        op = None
        dist = 0
        for p in self.anchors:
            if op:
                dist += Point.distance(op, p)
            op = p
        if(dist < self.lineMinDistance):
            return
        
        if(not self.fitCircle()):
            self.drawPath()
            
        self.anchors.clear()