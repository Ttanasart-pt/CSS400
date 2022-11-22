import cv2 as cv
import numpy as np
import math
from shapely.geometry import LineString, Point
import util.cvpainter as cvpainter


def deep_index(lst, w, width_ratio, height_ratio):
    return [(sub.index(w)*width_ratio, i*height_ratio) for (i, sub) in enumerate(lst) if w in sub]
def detectLines(frame):
    CANVAS_WIDTH = frame.shape[1]
    CANVAS_HEIGHT = frame.shape[0]
    RESIZE_WIDTH = 160
    RESIZE_HEIGHT = 160
    width_ratio = int(CANVAS_WIDTH / RESIZE_WIDTH)
    height_ratio = int(CANVAS_HEIGHT / RESIZE_HEIGHT)

    kernel = np.ones((5,5), np.uint8)
    opening = cv.morphologyEx(frame, cv.MORPH_OPEN, kernel)
    result = np.zeros_like(frame)
    resize = cv.resize(opening, (RESIZE_HEIGHT, RESIZE_WIDTH) , interpolation=cv.INTER_NEAREST)
    dst = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)

    lineList = deep_index(dst.tolist(), 255, width_ratio, height_ratio)
    if len(lineList) == 0:
        return result
    first_index = lineList[0]
    last_index = lineList[-1]

    cv.line(result, first_index, last_index, (255,0,0), 3, cv.LINE_AA)
    
    """cv.imshow("Source", frame)
    cv.imshow("Resize", resize)
    cv.waitKey()"""
    return result

def detectShape(frame):
    CANVAS_WIDTH = frame.shape[1]
    CANVAS_HEIGHT = frame.shape[0]
    RESIZE_WIDTH = 160
    RESIZE_HEIGHT = 160
    width_ratio = int(CANVAS_WIDTH / RESIZE_WIDTH)
    height_ratio = int(CANVAS_HEIGHT / RESIZE_HEIGHT)

    result = np.zeros_like(frame)
    resize = cv.resize(frame, (RESIZE_HEIGHT, RESIZE_WIDTH) , interpolation=cv.INTER_CUBIC)
    
    dst = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    linesP = cv.HoughLinesP(dst, 1, np.pi/180, 20, None, 20, 20)

    blur = cv.GaussianBlur(dst, (5, 5), 0)
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 20, param1=15, param2=15, minRadius=0, maxRadius=0)

    #Probability Hough Lines
    if linesP is not None:
        
        #Create an array contains LineString objects for each line
        linesArr = []

        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1 = l[0]*width_ratio
            y1 = l[1]*height_ratio
            x2 = l[2]*width_ratio
            y2 = l[3]*height_ratio

            if abs(l[0]-l[2]) < 10 and abs(l[1]-l[3]) > 10:
                x2 = x1
            if abs(l[1]-l[3]) < 10 and abs(l[0]-l[2]) > 10:
                y2 = y1

            pt1 = (x1, y1)
            pt2 = (x2, y2)
            linesArr.append(LineString([pt1, pt2]))

        #Remove close/similar lines    
        i = 0
        while i != len(linesArr):
            l1 = linesArr[i]
            j = i+1
            while j < len(linesArr):
                if l1.centroid.distance(linesArr[j].centroid) < 25:
                    linesArr.pop(j)
                    i=0
                    j=i+1
                j += 1
            i += 1

        #Create a list contains all points
        points = []
        for i in range(0, len(linesArr)):
            points.append(linesArr[i].coords[0])
            points.append(linesArr[i].coords[1])

        #Close the gap between the lines
        while(len(points) > 0):
            point = tuple(points.pop(0))
            dist = math.inf
            closestPointIndex = None

            for i in range(0, len(points)):
                if Point(point).distance(Point(points[i])) < dist:
                    dist = Point(point).distance(Point(points[i]))
                    closestPointIndex = i

            closestPoint = points.pop(closestPointIndex)
            lineStringIndex = None
            isFirst = False

            for i in linesArr:
                if i.coords[0] == point:
                    lineStringIndex = linesArr.index(i)
                    isFirst = True
                elif i.coords[1] == point:
                    lineStringIndex = linesArr.index(i)

            if isFirst:
                newLineString = LineString([closestPoint, linesArr[lineStringIndex].coords[1]])
            else:
                newLineString = LineString([linesArr[lineStringIndex].coords[0], closestPoint])
            linesArr[lineStringIndex] = newLineString

        for i in range(0, len(linesArr)):   
            cv.line(result, (int(linesArr[i].coords[0][0]), int(linesArr[i].coords[0][1])), (int(linesArr[i].coords[1][0]),
            int(linesArr[i].coords[1][1])), (255,0,0), 3, cv.LINE_AA)

    if circles is not None and linesP is None:
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            cv.circle(result, (i[0]*width_ratio, i[1]*height_ratio), i[2]*width_ratio, (0, 255, 0), 2)
            cv.circle(result, (i[0]*width_ratio, i[1]*height_ratio), 2, (0, 0, 255), 3)

    return result

def drawLines(drawnPoints, surface):
    op = None
    for p in drawnPoints:
        if op is None:
            op = p
            continue
        cvpainter.draw_line(surface, op, p, 5, (0, 255, 255))
        op = p
    return surface