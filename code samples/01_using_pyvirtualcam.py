import pyvirtualcam
import cv2

def main():
    camIn = cv2.VideoCapture(0)
    camIn.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camIn.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    with pyvirtualcam.Camera(width = 1280, height = 720, fps = 30) as camOut:
        print(f'Using virtual camera: {camOut.device}')
        
        while True:
            _, img = camIn.read()
            
            # do image processing here
            
            camOut.send(img)
            camOut.sleep_until_next_frame()
            
if __name__ == "__main__":
    main()