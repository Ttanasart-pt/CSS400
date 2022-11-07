import math

class Point():
    @staticmethod
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    @staticmethod
    def dot(p1, p2):
        return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z
    @staticmethod
    def distanceToLine(p, l1, l2):
        d = (l1 - l2).normalize()
        v = p - l2
        t = Point.dot(v, d)
        P = l2 + d * t
        return Point.distance(l1, P)
    
    def __init__(self, fromLandmark = None, fromTuple = None) -> None:
        if fromLandmark:
            self.x = fromLandmark.x
            self.y = fromLandmark.y
            self.z = fromLandmark.z
        if fromTuple:
            self.x = fromTuple[0]
            self.y = fromTuple[1]
            self.z = fromTuple[2]
    
    def normalize(self):
        return self / self.magnitude()
    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    def __add__(self, o):
        return Point(fromTuple = (self.x + o.x, self.y + o.y, self.z + o.z))
    def __sub__(self, o):
        return Point(fromTuple = (self.x - o.x, self.y - o.y, self.z - o.z))
    def __mul__(self, o):
        return Point(fromTuple = (self.x * o, self.y * o, self.z * o))
    def __truediv__(self, o):
        return Point(fromTuple = (self.x / o, self.y / o, self.z / o))
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"
    
def distancePointToLine(l0, l1, p):
    return abs((l1.x - l0.x) * (l0.x - p.y) - (l0.x - p.x) * (l1.y - l0.y)) / math.sqrt((l0.x - l0.y) ** 2 + (l1.x - l1.y) ** 2)