from unittest.mock import Mock
from unittest.mock import patch
import unittest
import person_tracker.utils as utils
from person_tracker.person import Person

utils.stack_images = Mock()


class TestUtils(unittest.TestCase):

    def test_extract_encoded_faces_with_empty_persons(self):
        persons = []
        self.assertEqual(len(utils.extract_encoded_faces(persons)), 0)

    def test_extract_encoded_faces_with_2_persons(self):
        persons = [Person(None, None, None, None), Person(None, None, None, None)]
        self.assertEqual(len(utils.extract_encoded_faces(persons)), 2)

    @patch('person_tracker.utils.cv2.imshow')
    @patch('person_tracker.utils.cv2.cvtColor')
    def test_show_images_with_empty_persons(self,cv2_cvtColor_mock, cv2_imshow_mock):
        persons = []
        utils.show_images(persons)

        self.assertEqual(cv2_imshow_mock.call_count, 0)
        self.assertEqual(cv2_cvtColor_mock.call_count, 0)

    @patch('person_tracker.utils.cv2.imshow')
    @patch('person_tracker.utils.cv2.cvtColor')
    def test_show_images(self, cv2_cvtColor_mock, cv2_imshow_mock):
        persons = [Person(None, None, None, None), Person(None, None, None, None)]
        utils.show_images(persons)

        self.assertEqual(cv2_imshow_mock.call_count, 1)
        self.assertEqual(cv2_cvtColor_mock.call_count, 2)


if __name__ == '__main__':
    unittest.main()
