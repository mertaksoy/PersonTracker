from camera import Camera
from logic import PersonTracker
import cv2
import utils

if __name__ == '__main__':
    persons = []
    cam = Camera(640, 480)
    personTracker = PersonTracker(persons)

    while True:
        image = cam.read(display=True)
        personTracker.track_persons(Camera.scale_down(image))
        print('persons:', len(persons))
        utils.show_images(persons)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
