from camera import Camera
import cv2

cam = Camera(640, 480)

if __name__ == '__main__':

    while True:
        cv2.imshow("Camera", cam.read())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
