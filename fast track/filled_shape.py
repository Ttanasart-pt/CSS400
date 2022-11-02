import cv2 as cv
import numpy as np
import scipy as sp
from scipy import ndimage
from PIL import Image


class FilledShape:
    def __init__(self, img):
        self.img = img
        self.drawnImg = np.zeros((480, 640, 3), dtype = np.uint8)

    def detect(self, contour, debug):
        shape = "undefined"
        epsilon = 0.03 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv.boundingRect(contour)
        x *= 15
        y *= 20
        w *= 15
        h *= 20
        #cv.rectangle(self.img, (self.x, self.y), (self.x + w, self.y + h), (0, 0, 255), 5)
        #cv.rectangle(self.img, (self.x, self.y-10), (self.x + w, self.y + 10), (0, 0, 255), -1)
        font = cv.FONT_HERSHEY_SIMPLEX
        number = ""
        if debug:
            cv.drawContours(self.img, [contour], 0, (0, 255, 0), 2)

            for pt in approx:
                cv.circle(self.img, (pt[0][0], pt[0][1]), 5, (255, 0, 0), -1)
            number = str(len(approx)) + " "
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # print(w, h, w / h)
            if 0.95 < w / h < 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
            cv.rectangle(self.drawnImg, (x, y), (x + w, y + h), (255, 0, 0), 3)
        elif len(approx) == 5:
            shape = "Pentagon"
            #cv.polylines(self.drawnImg, )
        elif len(approx) > 5:
            shape = "Circle"
            cv.circle(self.drawnImg, (int((x+x+w)/2), int((y+y+h)/2)), int(w/2), (255, 0 ,0), 3)
        print(number+shape)
        cv.putText(self.drawnImg, number + shape, (x, y), font, 0.4, (255, 255, 255), 1, cv.LINE_AA)

    def preprocessing_image(self):
        """ img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(img_gray, 127, 255, 0)
        kernel = np.ones((5, 5), np.uint8)
        cv.erode(threshold, kernel, iterations = 10)
        cv.dilate(threshold, kernel, iterations = 1)
        threshold = cv.GaussianBlur(threshold, (3, 3), 0)
        contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) """
        #
        img_gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        kernel = np.ones((4, 4), np.uint8)
        dilation = cv.dilate(img_gray, kernel, iterations=1)
        blur = cv.GaussianBlur(dilation, (5, 5), 0)
        kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0,-1, 0]])
        sharp = cv.filter2D(blur, -1, kernel2)
        _, threshold = cv.threshold(sharp, 127, 255, 0)
        contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        return threshold, contours


def capture(frame, debug=False):
    resized = cv.resize(frame, (32, 32) , interpolation=cv.INTER_CUBIC)
    img_object = FilledShape(resized)
    threshold, contours = img_object.preprocessing_image()
    for contour in  contours:
        img_object.detect(contour, debug)
    
    cv.imshow('Threshold', threshold)
    cv.imshow('Original', frame)
    cv.waitKey()
    return img_object.drawnImg