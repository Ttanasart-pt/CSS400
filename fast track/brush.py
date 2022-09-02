from geometry import Point
import cvpainter

class Stroke():
    def __init__(self, surface) -> None:
        self.anchors = []
        self.surface = surface
        self.thickness = 5
        self.color = (0, 255, 255)
        
        self.anchorMinDistance = 16
    
    def record(self, point):
        self.anchors.append(point)
    
    def release(self):
        if(len(self.anchors) == 0):
            return
        
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                continue
            if(Point.distance(op, p) < self.anchorMinDistance):
                continue
            
            cvpainter.draw_line(self.surface, op, p, self.thickness, self.color)
            op = p
            
        self.anchors.clear()