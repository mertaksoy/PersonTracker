import cv2


def extract_encoded_faces(persons):
    encoded_faces = []
    for person in persons:
        encoded_faces.append(person.face_encoding)
    return encoded_faces


def show_images(persons):
    for index in range(len(persons)):
        cv2.imshow("Person " + str(index), persons[index].image)
