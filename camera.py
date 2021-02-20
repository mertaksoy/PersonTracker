import cv2


class Camera:
    def __init__(self, width, height):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, width)
        self.capture.set(4, height)

    def read(self, display=False):
        success, img = self.capture.read()
        if display:
            cv2.imshow('Camera', img)
        return img

    @staticmethod
    def scale_down(img):
        img_scaled = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        return cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
