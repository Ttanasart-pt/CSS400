from util.geometry import Point
import util.cvpainter as painter
import cv2

class renderObject():
    def render():
        pass

class pathObject(renderObject) :
    def __init__(self, anchors) -> None:
        super().__init__()
        self.anchors = anchors
        self.thickness = 5
        self.color = (0, 255, 255)
        
    def render(self, render):
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                continue
            painter.draw_line(render, op, p, self.thickness, self.color)
            op = p