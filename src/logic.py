import face_recognition
from itertools import compress
import numpy as np
import src.person as person
import src.utils as utils


class PersonTracker:

    def __init__(self, persons):
        self.persons = persons

    def track_persons(self, scaled_image):
        face_locations, face_encodings = self.__detect_faces(scaled_image)
        for faceEncoding, faceLocation in zip(face_encodings, face_locations):
            y1, x2, y2, x1 = faceLocation
            if len(self.persons) == 0 and len(face_encodings) > 0:
                person.create(faceEncoding, scaled_image[y1:y2, x1:x2], self.persons)
            elif len(face_encodings) > 0:
                matches, match_index = self.__recognize_face(utils.extract_encoded_faces(self.persons), faceEncoding)
                if matches[match_index]:
                    self.persons[match_index].recognized()
                else:
                    person.create(faceEncoding, scaled_image[y1:y2, x1:x2], self.persons)
                self.__remove_similar_faces(matches)

    def __detect_faces(self, scaled_image):
        face_locations = face_recognition.face_locations(scaled_image)
        face_encodings = face_recognition.face_encodings(scaled_image, face_locations)
        return face_locations, face_encodings

    def __compare_faces(self, known_face_encodings, face_encoding):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_dis = face_recognition.face_distance(known_face_encodings, face_encoding)
        return matches, face_dis

    def __recognize_face(self, known_face_encodings, face_encoding):
        matches, face_dis = self.__compare_faces(known_face_encodings, face_encoding)
        return matches, np.argmin(face_dis)

    def __remove_similar_faces(self, face_matches):
        index_of_similar_faces = list(compress(range(len(face_matches)), face_matches))
        n_of_similar_faces = len(index_of_similar_faces)
        if n_of_similar_faces > 1:
            position_to_remove = index_of_similar_faces[n_of_similar_faces - 1]
            self.persons.remove(self.persons[position_to_remove])
            face_matches.remove(face_matches[position_to_remove])
            self.__remove_similar_faces(face_matches)
        else:
            return
