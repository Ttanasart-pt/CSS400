from util.geometry import Point, distancePointToLine
import util.renderObject as renderObject

class StrokeCapture():
    def __init__(self, renderer) -> None:
        self.anchors = []
        self.renderer = renderer
        self.thickness = 5
        self.color = (0, 255, 255)
        
        self.anchorMinDistance = 8
        self.lineMinDistance = 32
        self.nAverage = 3
        self.douglasThres = 4
        
        self.isDrawing = False
        self.smoothAlgo = 'Distance accumulate'
    
    def record(self, point):
        self.anchors.append(point)
        self.isDrawing = True
    
    def fitCircle(self):
        return False
    
    def drawPath(self):
        if self.smoothAlgo == 'Distance accumulate':
            self.drawPathDistAcc()
        elif self.smoothAlgo == 'Moving average':
            self.drawPathMoveAvg()
        elif self.smoothAlgo == 'Douglas-Peucker':
            self.drawPathDouglas()
    
    def drawPathDistAcc(self):
        anchors = []
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                anchors.append(p)
                continue
            if(Point.distance(op, p) < self.anchorMinDistance):
                continue
            
            anchors.append(p)
            op = p
        self.renderer.addObject(renderObject.pathObject(anchors))
    
    def drawPathMoveAvg(self):
        anchors = []
        pool = []
        
        for p in self.anchors:
            pool.append(p)
            if(len(pool) > self.nAverage):
                pool.pop(0)
            
            _p = Point(fromTuple = (0, 0, 0))
            for pl in pool:
                _p += pl
            
            avg = _p / len(pool)
            avg.x = round(avg.x)
            avg.y = round(avg.y)
            avg.z = round(avg.z)
            anchors.append(avg)
        self.renderer.addObject(renderObject.pathObject(anchors))
        
    def drawPathDouglas(self):
        anchors = []
        op = None
        oi = 0
        for i, p in enumerate(self.anchors):
            if op is None:
                op = p
                anchors.append(p)
                continue
            
            maxDist = 0
            for j in self.anchors[oi:i]:
                maxDist = max(maxDist, distancePointToLine(op, p, j))
            
            if(maxDist > self.douglasThres):
                anchors.append(p)
                op = p
                oi = i
        self.renderer.addObject(renderObject.pathObject(anchors))
        
    def release(self):
        self.isDrawing = False
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