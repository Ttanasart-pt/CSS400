import numpy as np

class Renderer():
    def __init__(self) -> None:
        self.objects = []
        
    def addObject(self, renderObject):
        self.objects.append(renderObject)
        
    def render(self, surface):
        surface = np.zeros_like(surface)
        for o in self.objects:
            o.render(surface)
        return surface
            
    def clear(self):
        self.objects.clear()