from util.geometry import Point
import util.cvpainter as painter
import cv2

class renderObject():
    def __init__(self) -> None:
        self.thickness = 5
        self.color = (0, 255, 255)
    
    def setStyle(self, **kwargs):
        if 'thickness' in kwargs:
            self.thickness = kwargs['thickness']
        if 'color' in kwargs:
            self.color = kwargs['color']
        return self
    
    def render():
        pass

class pathObject(renderObject) :
    def __init__(self, anchors) -> None:
        super().__init__()
        self.anchors = anchors
        
    def render(self, surface):
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                continue
            painter.draw_line(surface, op, p, self.thickness, self.color)
            op = p
            
class circleObject(renderObject):
    def __init__(self, meanrad) -> None:
        super().__init__()
        self.center, self.radius = meanrad
        self.center.x = round(self.center.x)
        self.center.y = round(self.center.y)
        self.radius = round(self.radius)
        
    def render(self, surface):
        cv2.circle(surface, (self.center.x, self.center.y), self.radius, self.color, self.thickness)


class rectObject(renderObject):
    def __init__(self, corners) -> None:
        super().__init__()
        self.corners = corners
        self.corners[0] = round(self.corners[0])
        self.corners[1] = round(self.corners[1])
        self.corners[2] = round(self.corners[2])
        self.corners[3] = round(self.corners[3])
        
    def render(self, surface):
        cv2.rectangle(surface, self.corner[:2], self.corners[2:], self.color, self.thickness)