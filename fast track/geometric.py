import math

class Point():
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
        
    