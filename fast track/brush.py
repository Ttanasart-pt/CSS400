import cvpainter

class Stroke():
    def __init__(self, surface) -> None:
        self.anchors = []
        self.surface = surface
        self.thickness = 4
        self.color = (255, 0, 0)
    
    def record(self, point):
        self.anchors.append(point)
    
    def release(self):
        if(len(self.anchors) == 0):
            return
            
        for i in range(self.anchors - 1):
            p0 = self.anchors[i]
            p1 = self.anchors[i + 1]
            
            cvpainter.draw_line(self.surface, p0, p1, self.thickness, self.color)
            
        self.anchors.clear()