from camera import Camera
from person import Person
import cv2
import face_recognition
from datetime import datetime
import numpy as np
import utils

cam = Camera(640, 480)
persons = []


def clean_persons(face_matches, face_dis):
    previous_match = False
    for index in range(len(face_matches)):
        if face_matches[index] and previous_match:
            similarity = abs(face_dis[index] - face_dis[index - 1])
            if similarity < 0.05:
                persons.remove(persons[index])
        previous_match = face_matches[index]


if __name__ == '__main__':

    while True:
        image = cam.read()
        cv2.imshow("Camera", image)
        scaledImage = Camera.scale_down(image)

        faceLocations = face_recognition.face_locations(scaledImage)
        faceEncodings = face_recognition.face_encodings(scaledImage, faceLocations)

        if len(persons) == 0 and len(faceEncodings) > 0:
            for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
                y1, x2, y2, x1 = faceLocation
                persons.append(Person(datetime.now(), faceEncoding, scaledImage[y1:y2, x1:x2]))
        elif len(faceEncodings) > 0:
            for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
                knownFaceEncodings = utils.extract_encoded_faces(persons)
                matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
                faceDis = face_recognition.face_distance(knownFaceEncodings, faceEncoding)

                clean_persons(matches, faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    print('matched')
                    # nothing to do yet
                else:
                    print('not matched')
                    y1, x2, y2, x1 = faceLocation
                    persons.append(Person(datetime.now(), faceEncoding, scaledImage[y1:y2, x1:x2]))

        print(len(persons))
        utils.show_images(persons)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
