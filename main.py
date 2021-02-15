from camera import Camera
from person import Person
import cv2
import face_recognition
from datetime import datetime
import numpy as np

cam = Camera(640, 480)
persons = []
knownFaceEncodings = []

if __name__ == '__main__':

    while True:
        image = cam.read()
        cv2.imshow("Camera", image)
        scaledImage = Camera.scale_down(image)

        faceLocations = face_recognition.face_locations(scaledImage)
        faceEncodings = face_recognition.face_encodings(scaledImage, faceLocations)

        if len(faceEncodings) == 0:
            continue

        if len(knownFaceEncodings) == 0 & len(persons) == 0:
            for faceEncoding in faceEncodings:
                persons.append(Person(datetime.now()))
                knownFaceEncodings.append(faceEncoding)
        else:
            for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
                matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
                faceDis = face_recognition.face_distance(knownFaceEncodings, faceEncoding)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    print('matched')
                    # nothing to do yet
                else:
                    print('not matched')
                    persons.append(Person(datetime.now()))
                    knownFaceEncodings.append(faceEncoding)

        print(len(persons))
        print(len(knownFaceEncodings))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
