import cv2


class Camera:
    def __init__(self, width, height):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, width)
        self.capture.set(4, height)

    def read(self):
        success, img = self.capture.read()
        return img
