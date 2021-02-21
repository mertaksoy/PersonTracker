import unittest
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import call
from person_tracker.camera import Camera
import cv2


class TestCamera(unittest.TestCase):

    @patch('person_tracker.camera.cv2.resize')
    @patch('person_tracker.camera.cv2.cvtColor')
    def test_scale_down(self, cv2_cvtColor_mock, cv2_resize_mock):
        image = Camera.scale_down(Mock())

        self.assertEqual(cv2_cvtColor_mock.call_count, 1)
        self.assertEqual(cv2_resize_mock.call_count, 1)
        self.assertIsNotNone(image)

    @patch('person_tracker.camera.cv2.resize')
    @patch('person_tracker.camera.cv2.cvtColor')
    def test_scale_down_resizing_parameters(self, cv2_cvtColor_mock, cv2_resize_mock):
        mocked_image = Mock()
        Camera.scale_down(mocked_image)

        args = cv2_resize_mock.call_args.args
        self.assertEqual(args, (mocked_image, (0, 0), None, 0.25, 0.25))

    @patch('person_tracker.camera.cv2.resize')
    @patch('person_tracker.camera.cv2.cvtColor')
    def test_scale_down_cvtColor_parameters(self, cv2_cvtColor_mock, cv2_resize_mock):
        mocked_image = Mock()
        cv2_resize_mock.return_value = mocked_image
        Camera.scale_down(mocked_image)

        args = cv2_cvtColor_mock.call_args.args
        self.assertEqual(args, (mocked_image, cv2.COLOR_BGR2RGB))

    @patch('person_tracker.camera.cv2.VideoCapture')
    def test_camera_initialization(self, video_capture):
        mocked_video_capture = Mock()
        video_capture.return_value = mocked_video_capture
        cam = Camera(640, 480)

        self.assertIsNotNone(cam)
        self.assertEqual(video_capture.call_args.args, (0,))
        self.assertEqual(mocked_video_capture.set.call_count, 2)
        self.assertEqual(mocked_video_capture.set.call_args_list, [call(3, 640), call(4, 480)])

    @patch('person_tracker.camera.cv2.VideoCapture')
    def test_camera_read_with_display_false(self, video_capture):
        mocked_video_capture = Mock()
        mocked_video_capture_read = Mock()
        mocked_video_capture.read.return_value = mocked_video_capture_read, mocked_video_capture_read
        video_capture.return_value = mocked_video_capture
        cam = Camera(640, 480)
        image = cam.read()

        self.assertEqual(image, mocked_video_capture_read)
        self.assertEqual(mocked_video_capture.read.call_count, 1)

    @patch('person_tracker.camera.cv2.VideoCapture')
    @patch('person_tracker.camera.cv2.imshow')
    def test_camera_read_with_display_true(self, cv2_imshow_mock, video_capture):
        mocked_video_capture = Mock()
        mocked_video_capture.read.return_value = Mock(), Mock()
        video_capture.return_value = mocked_video_capture
        Camera(640, 480).read(display=True)

        self.assertEqual(cv2_imshow_mock.call_count, 1)
        self.assertEqual(mocked_video_capture.read.call_count, 1)




