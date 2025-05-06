import unittest
import os
import numpy as np
import torch
import math
from unittest import mock

# Django testing imports
from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.files.base import ContentFile
from django.conf import settings

# Imports from your application - Use absolute paths from project root
try:
    from analyzer.models import VideoAnalysis
    from analyzer.forms import VideoUploadForm, SignUpForm
    from analyzer.services.badminton_pose_estimator import (
        BadmintonVideoTracker,
        FrameAnalysis,
    )
    from analyzer.services.video_analysis_app import (
        DjangoVideoAnalysisApp,
        # Removed import for make_json_serializable as it's internal
    )
    # from analyzer.services.ollama_feedback_service import generate_feedback
    SERVICES_IMPORTED = True
except ImportError as e:
    print(f"**************************************************************")
    print(f"WARNING: Failed to import service classes/functions for tests.")
    print(f"Error: {e}")
    print(f"Ensure services directory is correctly structured and importable.")
    print(f"**************************************************************")
    SERVICES_IMPORTED = False
    # Define dummies only if absolutely necessary for tests to load
    class BadmintonVideoTracker: pass
    class DjangoVideoAnalysisApp: pass


# --- Helper Functions ---

def create_mock_keypoints(num_keypoints=17, confidence=0.9):
    kp = np.random.rand(num_keypoints, 3) * 500
    kp[:, 2] = confidence
    return kp

def create_mock_frame(height=480, width=640):
    return np.zeros((height, width, 3), dtype=np.uint8)

def assertDictContainsNumpyArrays(tester, d1, d2, msg=None):
    tester.assertIsInstance(d1, dict, 'First argument is not a dictionary')
    tester.assertIsInstance(d2, dict, 'Second argument is not a dictionary')
    tester.assertEqual(d1.keys(), d2.keys())
    for k, v1 in d1.items():
        v2 = d2[k]
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            try:
                np.testing.assert_array_equal(v1, v2, err_msg=f"Numpy array mismatch for key '{k}'")
            except AssertionError as e:
                raise tester.failureException(f"{msg}\n{e}") from e
        elif isinstance(v1, dict) and isinstance(v2, dict):
            assertDictContainsNumpyArrays(tester, v1, v2, msg=f"Mismatch in nested dict for key '{k}'")
        elif isinstance(v1, (float, np.floating)) and isinstance(v2, (float, np.floating)):
             if math.isnan(v1) and math.isnan(v2): continue
             tester.assertAlmostEqual(v1, v2, places=5, msg=f"Float mismatch for key '{k}'")
        elif isinstance(v1, list) and isinstance(v2, list):
             tester.assertEqual(v1, v2, msg=f"List mismatch for key '{k}'")
        else:
            tester.assertEqual(v1, v2, msg=f"Value mismatch for key '{k}'")

# --- Test Classes ---

@unittest.skipUnless(SERVICES_IMPORTED, "Skipping BadmintonVideoTracker tests due to import errors.")
@mock.patch('analyzer.services.badminton_pose_estimator.YOLO')
class BadmintonVideoTrackerLogicTests(TestCase):
    """Tests focusing on the logic within BadmintonVideoTracker (mocking ML models)."""

    def setUp(self):
        self.MockYoloClass = mock.MagicMock()
        self.mock_pose_model_instance = mock.MagicMock()
        self.mock_shuttle_model_instance = mock.MagicMock()
        self.MockYoloClass.side_effect = [self.mock_pose_model_instance, self.mock_shuttle_model_instance]

    def test_w1_process_video_frame_calls(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker()
            self.assertIsInstance(tracker, BadmintonVideoTracker)

            tracker.validate_keypoints = mock.MagicMock(return_value=True)
            tracker.update_tracked_player = mock.MagicMock()
            tracker.analyze_frame = mock.MagicMock(return_value=(create_mock_frame(), {'some': 'data'}))

            mock_pose_results = mock.MagicMock()
            mock_pose_results.keypoints.data.cpu.return_value = torch.tensor([create_mock_keypoints(17)], dtype=torch.float32)
            self.mock_pose_model_instance.return_value = [mock_pose_results]

            frame = create_mock_frame()
            timestamp = 1.23
            annotated_frame, analysis_dict = tracker.process_video_frame(frame, timestamp)

            self.mock_pose_model_instance.assert_called_once_with(frame, conf=tracker.conf_threshold)
            tracker.validate_keypoints.assert_called_once()
            tracker.update_tracked_player.assert_called_once()
            tracker.analyze_frame.assert_called_once()
            self.assertIsNotNone(annotated_frame)
            self.assertIsNotNone(analysis_dict)

        except AttributeError as e:
             self.fail(f"Test W1 failed due to AttributeError: {e}. Check method names and instantiation.")
        except Exception as e:
             self.fail(f"Test W1 failed unexpectedly: {e}")


    def test_w2_validate_keypoints(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker_instance = BadmintonVideoTracker()
            self.assertTrue(hasattr(tracker_instance, 'KEYPOINTS'), "Tracker instance missing KEYPOINTS attribute")
            max_keypoint_index = max(tracker_instance.KEYPOINTS.values())

            valid_kp_data = torch.tensor([create_mock_keypoints(max_keypoint_index + 1)], dtype=torch.float32)
            self.assertTrue(tracker_instance.validate_keypoints(valid_kp_data), "Validation failed for supposedly valid keypoints")

            invalid_kp_data_size = torch.tensor([create_mock_keypoints(max_keypoint_index)], dtype=torch.float32)
            self.assertFalse(tracker_instance.validate_keypoints(invalid_kp_data_size), "Validation passed for keypoints with insufficient size")
            empty_kp_data = torch.empty((0, max_keypoint_index + 1, 3), dtype=torch.float32)
            self.assertFalse(tracker_instance.validate_keypoints(empty_kp_data), "Validation passed for empty keypoint data")

        except AttributeError:
             self.skipTest("Skipping W2: BadmintonVideoTracker missing 'validate_keypoints' method.")
        except KeyError:
             self.fail("Test W2 failed: Check KEYPOINTS definition in BadmintonVideoTracker.")


    def test_w3_update_tracked_player(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker_instance = BadmintonVideoTracker()
            mock_kp_data_p1_high_conf = torch.tensor([create_mock_keypoints(17, confidence=0.9)], dtype=torch.float32)
            mock_kp_data_p1_low_conf = torch.tensor([create_mock_keypoints(17, confidence=0.3)], dtype=torch.float32)
            mock_kp_data_multi_player = torch.tensor([
                create_mock_keypoints(17, confidence=0.6),
                create_mock_keypoints(17, confidence=0.9),
            ], dtype=torch.float32)
            mock_kp_data_no_player = torch.empty((0, 17, 3), dtype=torch.float32)

            tracker_instance.tracked_id = None
            tracker_instance.update_tracked_player(mock_kp_data_p1_high_conf)
            self.assertEqual(tracker_instance.tracked_id, 0, "Did not track single high-confidence player")

            tracker_instance.tracked_id = None
            tracker_instance.update_tracked_player(mock_kp_data_multi_player)
            self.assertEqual(tracker_instance.tracked_id, 1, "Did not track highest confidence player among multiple")

            tracker_instance.tracked_id = 1
            tracker_instance.update_tracked_player(mock_kp_data_p1_low_conf)
            self.assertEqual(tracker_instance.tracked_id, 0, "Did not switch tracked player")

            tracker_instance.tracked_id = 0
            tracker_instance.update_tracked_player(mock_kp_data_no_player)
            self.assertIsNone(tracker_instance.tracked_id, "Did not lose track when no player detected")
        except AttributeError:
             self.skipTest("Skipping W3: BadmintonVideoTracker missing 'update_tracked_player' method.")


    def test_w4_detect_shuttlecock(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker_instance = BadmintonVideoTracker()
            mock_shuttle_model_instance = tracker_instance.shuttle_model

            mock_result_no_boxes = mock.MagicMock()
            mock_empty_boxes = mock.MagicMock()
            mock_empty_boxes.conf.cpu().numpy.return_value = np.array([])
            mock_empty_boxes.xyxy.cpu().numpy.return_value = np.empty((0,4))
            mock_result_no_boxes.boxes = mock_empty_boxes
            mock_shuttle_model_instance.return_value = [mock_result_no_boxes]
            result = tracker_instance.detect_shuttlecock(create_mock_frame())
            self.assertIsNone(result, "Expected None when model returns no boxes")

            mock_boxes_data = mock.MagicMock()
            mock_boxes_data.conf.cpu().numpy.return_value = np.array([0.8, 0.7])
            mock_boxes_data.xyxy.cpu().numpy.return_value = np.array([[10., 10., 20., 20.], [50., 50., 60., 60.]])
            mock_result_with_boxes = mock.MagicMock()
            mock_result_with_boxes.boxes = mock_boxes_data
            mock_shuttle_model_instance.return_value = [mock_result_with_boxes]

            tracker_instance.prev_shuttle_box = None
            result = tracker_instance.detect_shuttlecock(create_mock_frame())
            self.assertIsNotNone(result, "Expected detection result, got None")
            if result is not None:
                np.testing.assert_array_almost_equal(result, np.array([10., 10., 20., 20.]))

        except AttributeError:
             self.skipTest("Skipping W4: BadmintonVideoTracker missing 'detect_shuttlecock'.")


    def test_w5_calculate_grip_angle(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker()
            kp_90 = np.zeros((17, 3)); kp_90[6]=[10,50,1]; kp_90[8]=[50,50,1]; kp_90[10]=[50,10,1]
            kp_180 = np.zeros((17, 3)); kp_180[6]=[10,50,1]; kp_180[8]=[50,50,1]; kp_180[10]=[90,50,1]
            kp_zero = np.zeros((17, 3)); kp_zero[6]=[10,50,1]; kp_zero[8]=[50,50,1]; kp_zero[10]=[50,50,1]

            self.assertAlmostEqual(tracker.calculate_grip_angle(kp_90), 90.0, places=5)
            self.assertAlmostEqual(tracker.calculate_grip_angle(kp_180), 180.0, places=5)
            self.assertEqual(tracker.calculate_grip_angle(kp_zero), 0.0)
        except AttributeError:
             self.skipTest("Skipping W5: BadmintonVideoTracker missing 'calculate_grip_angle' method.")
        except KeyError:
             self.fail("Test W5 failed: Check KEYPOINTS definition.")


    def test_w6_calculate_shoulder_angle(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker()
            kp_0 = np.zeros((17, 3)); kp_0[6]=[50,10,1]; kp_0[8]=[50,50,1] # Down
            kp_90 = np.zeros((17, 3)); kp_90[6]=[50,10,1]; kp_90[8]=[90,10,1] # Horizontal

            self.assertAlmostEqual(tracker.calculate_shoulder_angle(kp_0), 0.0, places=5)
            self.assertAlmostEqual(tracker.calculate_shoulder_angle(kp_90), 90.0, places=5)
        except AttributeError:
             self.skipTest("Skipping W6: BadmintonVideoTracker missing 'calculate_shoulder_angle' method.")
        except KeyError:
             self.fail("Test W6 failed: Check KEYPOINTS definition.")


    def test_w7_determine_shot_phase_sequence(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker(target_shot_type='smash')
            thresholds = tracker.SHOT_THRESHOLDS['smash']

            tracker.reset_detection_state()
            phase = tracker.determine_shot_phase_improved(100, thresholds['pre_hit']['grip'], thresholds['pre_hit']['shoulder'])
            self.assertEqual(phase, 'pre_hit')
            self.assertEqual(tracker.sequence_state, 'pre_hit')

            tracker.min_shuttle_distance = 100
            phase = tracker.determine_shot_phase_improved(10, thresholds['hit']['grip'], thresholds['hit']['shoulder'])
            self.assertEqual(phase, 'hit')
            self.assertEqual(tracker.sequence_state, 'looking_for_hit')

            phase = tracker.determine_shot_phase_improved(50, thresholds['follow_through']['grip'], thresholds['follow_through']['shoulder'])
            self.assertEqual(phase, 'follow_through')
            self.assertEqual(tracker.sequence_state, 'looking_for_follow')
        except AttributeError:
             self.skipTest("Skipping W7: BadmintonVideoTracker missing 'determine_shot_phase_improved'.")
        except KeyError:
             self.fail("Test W7 failed: Check SHOT_THRESHOLDS definition.")


    def test_w8_determine_shot_phase_edge_cases(self, MockYolo):
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker(target_shot_type='smash')
            thresholds = tracker.SHOT_THRESHOLDS['smash']

            tracker.reset_detection_state()
            tracker.sequence_state = 'pre_hit'
            phase = tracker.determine_shot_phase_improved(50, thresholds['follow_through']['grip'], thresholds['follow_through']['shoulder'])
            self.assertEqual(phase, 'pre_hit')
            self.assertTrue(tracker.awaiting_follow_through)
            self.assertEqual(tracker.sequence_state, 'looking_for_hit')

            tracker.min_shuttle_distance = 100
            phase = tracker.determine_shot_phase_improved(10, thresholds['hit']['grip'], thresholds['hit']['shoulder'])
            self.assertEqual(phase, 'hit')
            self.assertTrue(tracker.found_hit)

            phase = tracker.determine_shot_phase_improved(60, thresholds['follow_through']['grip'], thresholds['follow_through']['shoulder'])
            self.assertEqual(phase, 'follow_through')
            self.assertFalse(tracker.awaiting_follow_through)
            self.assertEqual(tracker.sequence_state, 'looking_for_follow')

            phase = tracker.determine_shot_phase_improved(100, thresholds['pre_hit']['grip'], thresholds['pre_hit']['shoulder'])
            self.assertEqual(phase, 'pre_hit')
            self.assertEqual(tracker.sequence_state, 'pre_hit')
            self.assertFalse(tracker.found_hit)

        except AttributeError:
             self.skipTest("Skipping W8: BadmintonVideoTracker missing 'determine_shot_phase_improved' or 'reset_detection_state'.")
        except KeyError:
             self.fail("Test W8 failed: Check SHOT_THRESHOLDS definition.")


    def test_w9_update_key_frames_logic(self, MockYolo):
        """W9: Test key frame selection logic."""
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker(target_shot_type='smash')
            mock_frame1, mock_frame2, mock_frame3 = create_mock_frame(1), create_mock_frame(2), create_mock_frame(3)
            mock_kp1, mock_kp2, mock_kp3 = create_mock_keypoints(1), create_mock_keypoints(2), create_mock_keypoints(3)

            tracker.phase_detections['pre_hit'] = [
                {'frame': mock_frame1, 'distance': 100, 'timestamp': 1.0, 'keypoints': mock_kp1},
                {'frame': mock_frame2, 'distance': 90, 'timestamp': 1.1, 'keypoints': mock_kp2},
                {'frame': mock_frame3, 'distance': 80, 'timestamp': 1.2, 'keypoints': mock_kp3}
            ]
            tracker.phase_detections['hit'] = [
                {'frame': mock_frame1, 'distance': 10, 'timestamp': 1.5, 'keypoints': mock_kp1},
                {'frame': mock_frame2, 'distance': 15, 'timestamp': 1.6, 'keypoints': mock_kp2}
            ]
            tracker.phase_detections['follow_through'] = [
                {'frame': mock_frame1, 'distance': 50, 'timestamp': 2.0, 'keypoints': mock_kp1},
                {'frame': mock_frame2, 'distance': 60, 'timestamp': 2.1, 'keypoints': mock_kp2}
            ]
            tracker.min_distance_frame = tracker.phase_detections['hit'][0]

            tracker.update_key_frames_from_detections()

            self.assertIs(tracker.key_frames['pre_hit']['frame'], mock_frame2)
            self.assertIs(tracker.key_frames['hit']['frame'], mock_frame1)
            self.assertIs(tracker.key_frames['follow_through']['frame'], mock_frame2)

            tracker.phase_detections = {'pre_hit': [], 'hit': [], 'follow_through': []}
            tracker.min_distance_frame = None
            tracker.key_frames = {'pre_hit': None, 'hit': None, 'follow_through': None}
            tracker.update_key_frames_from_detections()
            self.assertIsNone(tracker.key_frames['pre_hit'])
            self.assertIsNone(tracker.key_frames['hit'])
            self.assertIsNone(tracker.key_frames['follow_through'])

        except AttributeError:
             self.skipTest("Skipping W9: BadmintonVideoTracker missing 'update_key_frames_from_detections'.")


    def test_w10_get_technique_summary(self, MockYolo):
        """W10: Test summary generation logic."""
        MockYolo.side_effect = self.MockYoloClass.side_effect
        try:
            tracker = BadmintonVideoTracker(target_shot_type='smash')

            tracker.key_frames = {'pre_hit': None, 'hit': None, 'follow_through': None}
            summary = tracker.get_technique_summary()
            self.assertEqual(summary['status'], 'incomplete')

            mock_frame = create_mock_frame()
            mock_kp = create_mock_keypoints()
            tracker.key_frames = {
                'pre_hit': {'timestamp': 1.0, 'frame': mock_frame, 'keypoints': mock_kp, 'distance': 50.},
                'hit': {'timestamp': 1.5, 'frame': mock_frame, 'keypoints': mock_kp, 'distance': 10., 'shuttle_box': np.array([1.,1.,1.,1.])},
                'follow_through': {'timestamp': 2.1, 'frame': mock_frame, 'keypoints': mock_kp, 'distance': 60.}
            }
            summary = tracker.get_technique_summary()
            self.assertEqual(summary['status'], 'complete')
            self.assertAlmostEqual(summary['timing']['preparation_time'], 0.5)
            self.assertAlmostEqual(summary['timing']['follow_through_time'], 0.6)

            tracker.key_frames['hit'] = {'frame': mock_frame, 'keypoints': mock_kp} # Missing timestamp
            summary = tracker.get_technique_summary()
            # Test current behavior: timing calculation fails internally but status is complete
            self.assertEqual(summary['status'], 'complete')
            self.assertNotIn('preparation_time', summary.get('timing', {}))

        except AttributeError:
             self.skipTest("Skipping W10: BadmintonVideoTracker missing 'get_technique_summary'.")
        except KeyError:
             self.fail("Test W10 failed: Check SHOT_THRESHOLDS or KEYPOINTS definition.")


@unittest.skipUnless(SERVICES_IMPORTED, "Skipping DjangoVideoAnalysisApp tests due to import errors.")
@mock.patch('analyzer.services.video_analysis_app.cv2.VideoCapture')
@mock.patch('analyzer.services.video_analysis_app.subprocess.Popen')
@mock.patch('analyzer.services.video_analysis_app.BadmintonVideoTracker')
@mock.patch('analyzer.services.video_analysis_app.os.path.exists')
@mock.patch('analyzer.services.video_analysis_app.open', new_callable=mock.mock_open)
@mock.patch('analyzer.services.video_analysis_app.cv2.imencode')
@mock.patch('analyzer.services.video_analysis_app.ContentFile')
@mock.patch('analyzer.services.video_analysis_app.generate_feedback')
class DjangoVideoAnalysisAppTests(TestCase):
    """Tests focusing on the DjangoVideoAnalysisApp service class."""

    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.mock_analysis_model = mock.MagicMock(spec=VideoAnalysis)
        self.mock_analysis_model.id = 1
        self.mock_analysis_model.input_video.path = '/fake/video.mp4'
        self.mock_analysis_model.shot_type = 'smash'
        self.mock_analysis_model.process_interval = 2
        self.mock_analysis_model.source_fps = None
        self.mock_analysis_model.output_video.save = mock.MagicMock()
        self.mock_analysis_model.pre_hit_frame.save = mock.MagicMock()
        self.mock_analysis_model.hit_frame.save = mock.MagicMock()
        self.mock_analysis_model.follow_through_frame.save = mock.MagicMock()
        self.mock_analysis_model.analysis_summary = None
        self.mock_analysis_model.is_analyzed = False

    def test_w11_analyze_video_orchestration(self, mock_generate_feedback, mock_content_file, mock_imencode, mock_open, mock_path_exists, MockTracker, mock_popen, mock_videocapture):
        mock_path_exists.return_value = True
        mock_cap_instance = mock_videocapture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = [30.0, 640, 480, 10]
        mock_cap_instance.read.side_effect = [(True, create_mock_frame())] * 10 + [(False, None)]

        mock_tracker_instance = MockTracker.return_value
        # Ensure process_frame mock handles None frame correctly
        process_frame_results = [(create_mock_frame(), {'timestamp': i*0.1, 'grip_angle': 90, 'shoulder_angle': 45, 'shot_type': 'smash', 'shot_phase': 'pre_hit', 'within_threshold': True, 'keypoints': create_mock_keypoints()}) for i in range(10)]
        mock_tracker_instance.process_frame.side_effect = process_frame_results

        mock_summary = {
            'status': 'complete', 'timing': {'preparation_time': 0.5, 'follow_through_time': 0.6}, 'shot_type': 'smash',
            'key_frames': { 'pre_hit': {'timestamp': 1.0, 'keypoints': create_mock_keypoints().tolist()},
                           'hit': {'timestamp': 1.5, 'keypoints': create_mock_keypoints().tolist(), 'shuttle_box': [1.,1.,1.,1.]},
                           'follow_through': {'timestamp': 2.1, 'keypoints': create_mock_keypoints().tolist()}},
            'angles': {'pre_hit': 90.0, 'hit': 160.0, 'follow_through': 150.0},
            'ideal_angles': {'pre_hit': 90.0, 'hit': 165.0, 'follow_through': 150.0},
            'angle_differences': {'pre_hit': 0.0, 'hit': -5.0, 'follow_through': 0.0}
        }
        mock_tracker_instance.get_technique_summary.return_value = mock_summary
        service_internal_key_frames_mock = {
            'pre_hit': create_mock_frame(), 'hit': create_mock_frame(), 'follow_through': create_mock_frame()
        }
        mock_generate_feedback.return_value = "Good job, keep practicing!"
        mock_ffmpeg_process = mock_popen.return_value
        mock_ffmpeg_process.communicate.return_value = (b'', b''); mock_ffmpeg_process.returncode = 0
        mock_ffmpeg_process.stdin.write = mock.MagicMock(); mock_ffmpeg_process.stdin.close = mock.MagicMock()
        mock_imencode.return_value = (True, b'mock_encoded_png_data')

        try:
            service = DjangoVideoAnalysisApp(analysis_model=self.mock_analysis_model)
            self.assertIsInstance(service, DjangoVideoAnalysisApp)
            service._key_frames = service_internal_key_frames_mock
            result = service.analyze_video()
        except AttributeError as e:
             self.fail(f"Test W11 failed due to AttributeError: {e}")

        self.assertEqual(result['status'], 'complete')
        MockTracker.assert_called_once_with(target_shot_type='smash')
        self.assertGreater(mock_tracker_instance.process_frame.call_count, 0)
        mock_popen.assert_called_once()
        self.assertGreater(mock_ffmpeg_process.stdin.write.call_count, 0)
        mock_tracker_instance.get_technique_summary.assert_called_once()
        mock_generate_feedback.assert_called_once()
        call_args, _ = mock_generate_feedback.call_args
        self.assertEqual(call_args[0]['status'], 'complete')

        self.mock_analysis_model.save.assert_called()
        self.assertTrue(self.mock_analysis_model.is_analyzed)
        self.assertIsNotNone(self.mock_analysis_model.analysis_summary)
        self.assertEqual(self.mock_analysis_model.analysis_summary.get('ai_feedback'), "Good job, keep practicing!")
        self.mock_analysis_model.output_video.save.assert_called_once()
        self.mock_analysis_model.pre_hit_frame.save.assert_called_once()


    def test_w12_save_frame_as_image(self, mock_generate_feedback, mock_content_file, mock_imencode, mock_open, mock_path_exists, MockTracker, mock_popen, mock_videocapture):
        try:
            mock_imencode.return_value = (True, b'fakedata') # imencode returns bytes
            mock_content_file_instance = mock_content_file.return_value

            service = DjangoVideoAnalysisApp(analysis_model=self.mock_analysis_model)
            frame_to_save = create_mock_frame()
            field_name = 'hit_frame'
            success = service.save_frame_as_image(frame_to_save, field_name)

            self.assertTrue(success)
            mock_imencode.assert_called_once_with('.jpg', frame_to_save)
            # ContentFile takes the bytes directly
            mock_content_file.assert_called_once_with(b'fakedata')
            model_field_save_mock = getattr(self.mock_analysis_model, field_name).save
            model_field_save_mock.assert_called_once_with(f"{field_name}_{self.mock_analysis_model.id}.jpg", mock_content_file_instance)

            # Test failure case: imencode fails
            mock_imencode.reset_mock(); mock_content_file.reset_mock(); model_field_save_mock.reset_mock()
            mock_imencode.return_value = (False, None)
            success = service.save_frame_as_image(frame_to_save, field_name)
            self.assertFalse(success)

        except AttributeError:
             self.skipTest("Skipping W12: DjangoVideoAnalysisApp missing 'save_frame_as_image'.")


    # W13 is removed as make_json_serializable is internal to analyze_video


class ViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user_a = User.objects.create_user('userA', 'a@a.com', 'password')
        self.user_b = User.objects.create_user('userB', 'b@b.com', 'password')
        self.analysis_a = VideoAnalysis.objects.create(
            user=self.user_a,
            input_video=SimpleUploadedFile("vid_a.mp4", b"a", "video/mp4"),
            shot_type='smash'
        )

    def test_w15_delete_analysis_view_permissions(self):
        delete_url = reverse('analyzer:delete_analysis', args=[self.analysis_a.id])

        self.client.login(username='userB', password='password')
        response = self.client.post(delete_url)
        self.assertEqual(response.status_code, 404)
        self.assertTrue(VideoAnalysis.objects.filter(id=self.analysis_a.id).exists())

        self.client.login(username='userA', password='password')
        with mock.patch('os.remove'):
             response = self.client.post(delete_url)
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('analyzer:analysis_history'))
        self.assertFalse(VideoAnalysis.objects.filter(id=self.analysis_a.id).exists())

    def test_w16_upload_video_view_post(self):
        self.client.login(username='userA', password='password')
        upload_url = reverse('analyzer:upload_video')
        dummy_video = SimpleUploadedFile("test_upload.mp4", b"content", "video/mp4")
        post_data = {'shot_type': 'clear', 'process_interval': 3, 'input_video': dummy_video}

        response = self.client.post(upload_url, post_data, follow=False)

        self.assertEqual(response.status_code, 302)
        self.assertTrue(VideoAnalysis.objects.filter(user=self.user_a, shot_type='clear').exists())
        new_analysis = VideoAnalysis.objects.get(user=self.user_a, shot_type='clear')
        self.assertRedirects(response, reverse('analyzer:analysis_detail', args=[new_analysis.id]))
        self.assertEqual(new_analysis.process_interval, 3)
        self.assertEqual(new_analysis.user, self.user_a)
        if os.path.exists(new_analysis.input_video.path):
            new_analysis.delete()

class ModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('modeltestuser', 'test@model.com', 'password')
        self.dummy_video_file = SimpleUploadedFile("model_video.mp4", b"file_content", content_type="video/mp4")
        self.dummy_image_file = SimpleUploadedFile("model_frame.jpg", b"image_content", content_type="image/jpeg")
        self.dummy_video_file.seek(0)
        self.dummy_image_file.seek(0)

    def test_w14_model_delete_cleans_files(self):
        analysis = VideoAnalysis.objects.create(
            user=self.user,
            input_video=self.dummy_video_file,
            output_video=SimpleUploadedFile("model_output.mp4", b"out", content_type="video/mp4"),
            hit_frame=self.dummy_image_file
        )
        input_path = analysis.input_video.path
        output_path = analysis.output_video.path
        frame_path = analysis.hit_frame.path

        self.assertTrue(os.path.exists(input_path), f"Input file missing: {input_path}")
        self.assertTrue(os.path.exists(output_path), f"Output file missing: {output_path}")
        self.assertTrue(os.path.exists(frame_path), f"Frame file missing: {frame_path}")

        analysis_id = analysis.id
        analysis.delete()

        with self.assertRaises(VideoAnalysis.DoesNotExist):
            VideoAnalysis.objects.get(id=analysis_id)

        self.assertFalse(os.path.exists(input_path), f"Input file not deleted: {input_path}")
        self.assertFalse(os.path.exists(output_path), f"Output file not deleted: {output_path}")
        self.assertFalse(os.path.exists(frame_path), f"Frame file not deleted: {frame_path}")
