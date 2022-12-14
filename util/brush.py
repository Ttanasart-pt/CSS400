from util.geometry import Point, distancePointToLine, direction, distance
import util.renderObject as renderObject 

class StrokeCapture():
    def __init__(self, renderer) -> None:
        self.anchors = []
        self.renderer = renderer
        self.thickness = 4
        self.color = (0, 0, 255)
        
        self.anchorMinDistance = 8
        self.lineMinDistance = 32
        self.nAverage = 3
        self.douglasThres = 4
        self.shapeThres = 64
        
        self.isDrawing = False
        self.smoothAlgo = 'Distance accumulate'
    
    def record(self, point):
        self.anchors.append(point)
        self.isDrawing = True
    
    def fitCircle(self, thres):
        if(len(self.anchors) < 1):
            return False
         
        mean = Point()
        for p in self.anchors:
            mean += p
        mean /= len(self.anchors)
        
        dirr = 0
        op = None
        minx =  999999
        miny =  999999
        maxx = -999999
        maxy = -999999
        
        for p in self.anchors:
            if op is None:
                op = p
                continue
            minx = min(minx, p.x)
            miny = min(miny, p.y)
            maxx = max(maxx, p.x)
            maxy = max(maxy, p.y)
            dirr += direction(op, p)
        
        if(abs(180 - abs(dirr) % 360) > thres):
            return None
        
        rad = ((maxx - minx) + (maxy - miny)) / 2 / 2
        for p in self.anchors:
            dist = distance(p, mean)
            #if (abs(dist / rad - 1) > 0.4):
            #    return None
        return mean, rad
    
    def fitRectangle(self, thres):
        smoothAnchor = self.smoothPathDouglas(thres)
        if(len(smoothAnchor) != 4):
            return None
        
        dirr = 0
        op = None
        minx = None
        miny = None
        maxx = None
        maxy = None
        
        for p in self.anchors:
            if op is None:
                op = p
                minx = p.x
                miny = p.y
                maxx = p.x
                maxy = p.y
                continue
            dirr += direction(op, p)
            
        if(abs(180 - abs(dirr) % 360) > 30):
            return None
        return (minx, miny, maxx, maxy)
    
    def fitShape(self):
        shape = self.fitCircle(self.shapeThres)
        if(shape is not None):
            self.drawCircle(shape)
            return True
        
        shape = self.fitRectangle(self.shapeThres)
        if(shape is not None):
            self.drawRectangle(shape)
            return True
        
        return False
    
    def drawPath(self):
        smoothAnchor = None
        if self.smoothAlgo == 'Distance accumulate':
            smoothAnchor = self.smoothPathDistAcc(self.anchorMinDistance)
        elif self.smoothAlgo == 'Moving averae':
            smoothAnchor = self.smoothPathMoveAvg(self.nAverage)
        elif self.smoothAlgo == 'Douglas-Peucker':
            smoothAnchor = self.smoothPathDouglas(self.douglasThres)
            
        if smoothAnchor:
            self.renderer.addObject(self.applyStyle(renderObject.pathObject(smoothAnchor)))
    
    def drawCircle(self, meanRad):
        self.renderer.addObject(self.applyStyle(renderObject.circleObject(meanRad)))
    
    def drawRectangle(self, corners):
        self.renderer.addObject(self.applyStyle(renderObject.rectObject(corners)))
        
    def applyStyle(self, renderObj):
        return renderObj.setStyle(thickness = self.thickness, color = self.color)
    
    def smoothPathDistAcc(self, minDist):
        anchors = []
        op = None
        for p in self.anchors:
            if op is None:
                op = p
                anchors.append(p)
                continue
            if(Point.distance(op, p) < minDist):
                continue
            
            anchors.append(p)
            op = p
        return anchors
    
    def smoothPathMoveAvg(self, n):
        anchors = []
        pool = []
        
        for p in self.anchors:
            pool.append(p)
            if(len(pool) > n):
                pool.pop(0)
            
            _p = Point(fromTuple = (0, 0, 0))
            for pl in pool:
                _p += pl
            
            avg = _p / len(pool)
            avg.x = round(avg.x)
            avg.y = round(avg.y)
            avg.z = round(avg.z)
            anchors.append(avg)
        return anchors
        
    def smoothPathDouglas(self, threshold):
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
            
            if(maxDist > threshold):
                anchors.append(p)
                op = p
                oi = i
        return anchors
        
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
        
        if not self.fitShape():
            self.drawPath()
        
        self.anchors.clear()