import cv2

def draw_line(surf, p0, p1, thick, color):
    cv2.circle(surf, (p0.x, p0.y), thick - 1, color, -1)
    cv2.circle(surf, (p1.x, p1.y), thick - 1, color, -1)
    cv2.line(surf, (p0.x, p0.y), (p1.x, p1.y), color, thick)