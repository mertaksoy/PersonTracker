import unittest
from unittest.mock import Mock
from unittest.mock import patch
from person_tracker.logic import PersonTracker
import numpy as np


class TestUtils(unittest.TestCase):

    @patch('person_tracker.logic.face_recognition.face_locations')
    @patch('person_tracker.logic.face_recognition.face_encodings')
    def test_track_person_one_person_detected(self, face_encodings_mock, face_locations_mock):
        face_encodings_mock.return_value = [Mock()]
        face_locations_mock.return_value = [(1, 1, 2, 2)]

        person_tracker = PersonTracker([])
        person_tracker.track_persons(np.array([[1, 2, 3], [4, 5, 6]], np.int32))
        self.assertEqual(len(person_tracker.persons), 1)

    @patch('person_tracker.logic.face_recognition.face_distance')
    @patch('person_tracker.logic.face_recognition.compare_faces')
    @patch('person_tracker.logic.face_recognition.face_locations')
    @patch('person_tracker.logic.face_recognition.face_encodings')
    def test_track_person_two_person_detected_one_of_them_recognized(self, face_encodings_mock, face_locations_mock,
                                                                     compare_faces_mock,
                                                                     face_distance_mock):
        face_encodings_mock.return_value = [Mock(), Mock()]
        face_locations_mock.return_value = [(1, 1, 2, 2), (2, 2, 3, 3)]
        compare_faces_mock.return_value = [True, False]
        face_distance_mock.return_value = [0.01, 0.9]

        person_tracker = PersonTracker([])
        person_tracker.track_persons(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], np.int32))
        self.assertEqual(len(person_tracker.persons), 1)

    @patch('person_tracker.logic.face_recognition.face_distance')
    @patch('person_tracker.logic.face_recognition.compare_faces')
    @patch('person_tracker.logic.face_recognition.face_locations')
    @patch('person_tracker.logic.face_recognition.face_encodings')
    def test_track_person_two_new_person_detected(self, face_encodings_mock, face_locations_mock,
                                                  compare_faces_mock,
                                                  face_distance_mock):
        face_encodings_mock.return_value = [Mock(), Mock()]
        face_locations_mock.return_value = [(1, 1, 2, 2), (2, 2, 3, 3)]
        compare_faces_mock.return_value = [False, False]
        face_distance_mock.return_value = [0.9, 0.9]

        person_tracker = PersonTracker([])
        person_tracker.track_persons(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], np.int32))
        self.assertEqual(len(person_tracker.persons), 2)

    @patch('person_tracker.logic.face_recognition.face_distance')
    @patch('person_tracker.logic.face_recognition.compare_faces')
    @patch('person_tracker.logic.face_recognition.face_locations')
    @patch('person_tracker.logic.face_recognition.face_encodings')
    def test_track_person_remove_similar_faces(self, face_encodings_mock, face_locations_mock,
                                               compare_faces_mock,
                                               face_distance_mock):
        face_encodings_mock.return_value = [Mock()]
        face_locations_mock.return_value = [(1, 1, 2, 2)]
        compare_faces_mock.return_value = [True, True, False, False]
        face_distance_mock.return_value = [0.01, 0.02, 1, 1.4]

        person_tracker = PersonTracker([Mock(), Mock(), Mock(), Mock()])
        person_tracker.track_persons(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], np.int32))
        self.assertEqual(len(person_tracker.persons), 3)
