from datetime import datetime


def create(face_encoding, image, persons):
    persons.append(Person(datetime.now(), None, face_encoding, image))
    return persons


class Person:
    def __init__(self, first_seen, last_seen, face_encoding, image):
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.face_encoding = face_encoding
        self.image = image

    def recognized(self):
        self.last_seen = datetime.now()
