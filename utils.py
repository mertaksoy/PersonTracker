
def extract_encoded_faces(persons):
    encoded_faces = []
    for person in persons:
        encoded_faces.append(person.face_encoding)
    return encoded_faces
