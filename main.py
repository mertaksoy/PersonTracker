from camera import Camera
from person import Person
import cv2
import face_recognition
from datetime import datetime
import numpy as np
import utils
from itertools import compress

cam = Camera(640, 480)
persons = []


def remove_similar_faces(face_matches):
    index_of_similar_faces = list(compress(range(len(face_matches)), face_matches))
    n_of_similar_faces = len(index_of_similar_faces)
    if n_of_similar_faces > 1:
        position_to_remove = index_of_similar_faces[n_of_similar_faces - 1]
        persons.remove(persons[position_to_remove])
        face_matches.remove(face_matches[position_to_remove])
        remove_similar_faces(face_matches)
    else:
        return


if __name__ == '__main__':

    while True:
        image = cam.read(display=True)
        scaledImage = Camera.scale_down(image)

        faceLocations = face_recognition.face_locations(scaledImage)
        faceEncodings = face_recognition.face_encodings(scaledImage, faceLocations)

        for faceEncoding, faceLocation in zip(faceEncodings, faceLocations):
            y1, x2, y2, x1 = faceLocation
            if len(persons) == 0 and len(faceEncodings) > 0:
                persons.append(Person(datetime.now(), None, faceEncoding, scaledImage[y1:y2, x1:x2]))
            elif len(faceEncodings) > 0:
                knownFaceEncodings = utils.extract_encoded_faces(persons)
                matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)
                faceDis = face_recognition.face_distance(knownFaceEncodings, faceEncoding)

                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    persons[matchIndex].last_seen = datetime.now()
                else:
                    persons.append(Person(datetime.now(), None, faceEncoding, scaledImage[y1:y2, x1:x2]))
                remove_similar_faces(matches)

        print('persons: ' + str(len(persons)))
        utils.show_images(persons)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
