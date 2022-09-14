import cv2

def drawPointArray(img, landmark):
    points = []
    for i in range(int(len(landmark) / 2)):
        c = (round(landmark[i * 2]), round(landmark[i * 2 + 1]))
        img = cv2.circle(img, c, 2, (255, 0, 0), -1)
        points.append(c)
    return img, points