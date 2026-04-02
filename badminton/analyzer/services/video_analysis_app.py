"""
VideoAnalysisApp - Adapted for Django integration with AV1 codec support.
Handles video processing and badminton technique analysis.
"""
import json
import os
import subprocess
import time
import cv2
import numpy as np

from typing import Dict, Optional, Any, Tuple
from django.conf import settings
from django.utils import timezone
from django.core.files.base import ContentFile
from ..models import AnalysisJob
from .badminton_pose_estimator import BadmintonVideoTracker
from .ollama_feedback_service import generate_feedback # Import the feedback service


class DjangoVideoAnalysisApp:
    # Django-specific implementation of the video analysis application
    
    def __init__(self, analysis_model=None, analysis_job=None):
        """
        Initialize the analyzer with an optional analysis model instance.
        
        Args:
            analysis_model: Django model instance representing the analysis
        """
        # Django model for this analysis
        self.last_processed_frame: Any = None
        self.last_analysis: Any = None
        self.analysis_model = analysis_model
        self.analysis_job = analysis_job
        if self.analysis_job is None and self.analysis_model is not None:
            try:
                self.analysis_job = self.analysis_model.job
            except AnalysisJob.DoesNotExist:
                self.analysis_job = None
        
        # Core components
        self.tracker: Any = None
        self.current_video_path: Optional[str] = None
        self.analysis_summary: Any = None
        self.output_video_path: Optional[str] = None
        
        # Analysis state
        self._start_time: Optional[float] = None
        self._frame_count = 0
        self._is_analyzing = False
        self.stop_requested = False
        
        # Progress tracking for async feedback
        self._progress = 0
        self._status_message = "Initializing..."
        
        # Processing settings
        self._process_interval = 2  # Process every Nth frame
        self._source_fps = None  # Source video frame rate (None means auto-detect)
        
        # Key frames from shot analysis
        self._key_frames: Dict[str, Optional[np.ndarray]] = {
            'pre_hit': None,
            'hit': None,
            'follow_through': None
        }
    
    @property
    def progress(self) -> int:
        # Return current progress percentage
        return self._progress

    @progress.setter
    def progress(self, value: int) -> None:
        self._progress = max(0, min(100, int(value)))
        self._sync_job_state(progress=self._progress)
    
    @property
    def status_message(self) -> str:
        # Return current status message
        return self._status_message

    @status_message.setter
    def status_message(self, value: str) -> None:
        self._status_message = str(value)
        self._sync_job_state(message=self._status_message)
    
    @property
    def pre_hit_frame(self) -> Optional[np.ndarray]:
        return self._key_frames['pre_hit']
    
    @property
    def hit_frame(self) -> Optional[np.ndarray]:
        return self._key_frames['hit']
    
    @property
    def follow_through_frame(self) -> Optional[np.ndarray]:
        return self._key_frames['follow_through']
    
    def request_stop(self) -> str:
        # Request analysis process to stop gracefully
        if not self._is_analyzing:
            return "No analysis is currently running"
        
        self.stop_requested = True
        return "Stop requested. Analysis will finish soon..."

    def _sync_job_state(self, **updates) -> None:
        """Persist progress and status updates to the related job record."""
        if self.analysis_job is None or not updates:
            return

        try:
            for field_name, value in updates.items():
                setattr(self.analysis_job, field_name, value)
            self.analysis_job.save(update_fields=list(updates.keys()) + ['updated_at'])
        except Exception as e:
            print(f"Error syncing analysis job state: {e}")

    def _analysis_should_stop(self) -> bool:
        """Check whether a stop has been requested for this analysis."""
        if self.stop_requested:
            return True

        if self.analysis_job is None:
            return False

        try:
            self.analysis_job.refresh_from_db(fields=['cancel_requested'])
            return self.analysis_job.cancel_requested
        except Exception:
            return self.stop_requested

    @staticmethod
    def _buffer_to_bytes(buffer) -> bytes:
        """Normalize OpenCV encoder output into raw bytes."""
        if isinstance(buffer, bytes):
            return buffer
        return buffer.tobytes()
    
    def save_frame_as_image(self, frame: np.ndarray, field_name: str) -> bool:
        """
        Save a frame as an image in the analysis model.
        
        Args:
            frame: NumPy array representing the image
            field_name: Name of the model field to save to
            
        Returns:
            bool: True if save was successful
        """
        if frame is None or self.analysis_model is None:
            return False
        
        try:
            # Convert frame to JPEG image
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return False
            
            # Create a ContentFile from the buffer
            if isinstance(buffer, bytes):
                image_bytes = buffer
            else:
                image_bytes = buffer.tobytes()
            image_content = ContentFile(image_bytes)
            
            # Save to the specified field
            field = getattr(self.analysis_model, field_name)
            field.save(f"{field_name}_{self.analysis_model.id}.jpg", image_content)
            
            return True
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
            return False
    
    def update_key_frames(self, new_frames: Dict[str, Optional[Dict]]) -> None:
        # Safely update stored key frames from tracker results
        # Check each phase (pre-hit, hit, follow-through)
        for phase in self._key_frames:
            # Only process if we have data for this phase
            phase_data = new_frames.get(phase)
            if phase_data is not None:
                # Verify frame data exists before assignment
                frame_data = phase_data.get('frame')
                if isinstance(frame_data, np.ndarray):
                    self._key_frames[phase] = frame_data.copy()
    
    def initialize_tracker(self, shot_type: str) -> str:
        # Set up a new tracker instance for the specified shot type
        self.tracker = BadmintonVideoTracker(target_shot_type=shot_type)
        self._frame_count = 0
        self.stop_requested = False
        return f"Initialized tracker for {shot_type.replace('_', ' ').title()} analysis"
    
    def process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        # Process a single video frame for pose analysis
        if frame is None:
            return None, None
        
        try:
            # Use frame skipping logic in our application layer
            if hasattr(self, 'last_processed_frame') and hasattr(self, 'last_analysis'):
                if self._frame_count % self._process_interval != 0:
                    # Return previous results for skipped frames
                    if self.last_processed_frame is not None:
                        return self.last_processed_frame.copy(), self.last_analysis
                    return frame.copy(), self.last_analysis
            
            # Process the frame with the tracker
            tracker = self.tracker
            if hasattr(tracker, 'process_frame'):
                annotated_frame, analysis = tracker.process_frame(frame, current_time, self._frame_count)
            else:
                annotated_frame, analysis = tracker.process_video_frame(frame, current_time, self._frame_count)
            
            # Store results for potential frame skipping
            self.last_processed_frame = annotated_frame if annotated_frame is not None else frame.copy()
            self.last_analysis = analysis
            
            if analysis:
                # Update key frames based on the current tracker state
                self.update_key_frames(self.tracker.key_frames)
            
            return annotated_frame, analysis
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return frame, None

    @staticmethod
    def prepare_empty_frame() -> np.ndarray:
        # Create an empty frame with awaiting detection message
        empty_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        empty_frame.fill(25)  # Dark gray
        text = "Awaiting detection..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 0.7, 2)[0]
        
        # Center the text
        text_x = (empty_frame.shape[1] - textsize[0]) // 2
        text_y = (empty_frame.shape[0] + textsize[1]) // 2
        
        # Add text to empty frame
        cv2.putText(
            empty_frame,
            text,
            (text_x, text_y),
            font,
            0.7,
            (200, 200, 200),
            2
        )
        
        return empty_frame

    @staticmethod
    def generate_frame_report(analysis: Optional[Dict]) -> str:
        # Generate analysis report for the current frame
        if not analysis:
            return "No pose detected in this frame"

        return (
            f"🏸 Frame Analysis @ {analysis['timestamp']:.2f}s\n"
            f"Phase: {analysis['shot_phase'] or 'Preparing'}\n"
            f"Grip Angle: {analysis['grip_angle']:.1f}° | "
            f"Shoulder Angle: {analysis['shoulder_angle']:.1f}°\n"
            f"Form: {'✅ Within Threshold' if analysis['within_threshold'] else '❌ Needs Improvement'}"
        )

    def generate_summary_report(self) -> str:
        """Generate end-of-analysis summary report with timing details"""
        if not self.analysis_summary:
            return "⚠️ No analysis data available"
        
        if self.analysis_summary['status'] == 'incomplete':
            return (
                "⚠️ Incomplete Analysis\n"
                "Failed to capture all required phases.\n"
                "Ensure complete execution of:\n"
                "- Preparation\n"
                "- Hit\n"
                "- Follow-through\n\n"
                "Try adjusting the frame skip interval or using a different video."
            )
        
        timing = self.analysis_summary['timing']
        shot_type = self.analysis_summary['shot_type'].replace('_', ' ').title()
        
        return (
            f"✅ Analysis Complete: {shot_type} Technique\n\n"
            "⏱ Timing Analysis\n"
            f"- Preparation Time: {timing['preparation_time']:.3f}s\n"
            f"- Follow Through Time: {timing['follow_through_time']:.3f}s\n"
            f"- Total Shot Duration: {timing['preparation_time'] + timing['follow_through_time']:.3f}s\n\n"
            "The key frames shown above highlight your form at critical phases of the shot."
        )
    
    def save_key_frames(self) -> Dict[str, bool]:
        """Save key frames to the Django model."""
        if self.analysis_model is None:
            return {'pre_hit': False, 'hit': False, 'follow_through': False}
        
        results = {}
        
        # Map internal frame names to model field names
        field_mapping = {
            'pre_hit': 'pre_hit_frame',
            'hit': 'hit_frame',
            'follow_through': 'follow_through_frame'
        }
        
        for phase, field_name in field_mapping.items():
            frame = self._key_frames[phase]
            
            # Verify frame is valid before trying to save
            if frame is not None and isinstance(frame, np.ndarray):
                try:
                    results[phase] = self.save_frame_as_image(frame, field_name)
                except Exception as e:
                    print(f"Error saving {phase} frame: {str(e)}")
                    results[phase] = False
            else:
                results[phase] = False
        
        return results
    
    def analyze_video(self) -> Dict[str, Any]:
        # Ensure we have a valid model and video
        if self.analysis_model is None:
            return {
                'status': 'error',
                'message': 'No analysis model provided'
            }
        
        if not self.analysis_model.input_video:
            return {
                'status': 'error',
                'message': 'No input video provided'
            }
        
        # Get video path from model
        video_path = self.analysis_model.input_video.path
        
        # Get analysis parameters from model
        shot_type = self.analysis_model.shot_type
        process_interval = self.analysis_model.process_interval
        source_fps = self.analysis_model.source_fps
        
        # Setup for analysis
        process_interval = max(1, int(process_interval))  # Ensure valid interval
        self._process_interval = process_interval
        self.initialize_tracker(shot_type)
        self._source_fps = source_fps  # Store specified frame rate
        self._is_analyzing = True
        
        # Update status
        self.status_message = "Starting analysis..."
        self.progress = 0
        self._sync_job_state(
            status=AnalysisJob.STATUS_RUNNING,
            cancel_requested=False,
            started_at=timezone.now()
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Larger buffer for smoother reading
        
        if not cap.isOpened():
            self._is_analyzing = False
            self.status_message = "Error opening video file"
            self._sync_job_state(
                status=AnalysisJob.STATUS_FAILED,
                error_message='Could not open video file',
                finished_at=timezone.now()
            )
            return {
                'status': 'error',
                'message': 'Could not open video file'
            }
        
        # Get video properties
        auto_detected_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = self._source_fps if self._source_fps is not None else auto_detected_fps
        if fps <= 0:  # Fallback if invalid
            fps = 30
            print("Warning: Invalid FPS detected, using 30 FPS as default")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output directory and video writer
        media_root = settings.MEDIA_ROOT
        output_dir = os.path.join(media_root, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        output_filename = f"analysis_{timestamp}.mp4"  # Use .mp4 for direct download
        self.output_video_path = os.path.join(output_dir, output_filename)
        
        # Update status message
        self.status_message = f"Processing video ({total_frames} frames) at {fps:.1f} FPS, analyzing every {self._process_interval} frame"
        
        self._start_time = time.time()
        
        try:
            # Read all frames at once for more efficient processing
            self.status_message = "Reading video frames..."
            all_frames = []
            frame_count = 0
            analysis_canceled = False
            
            while cap.isOpened():
                if self._analysis_should_stop():
                    analysis_canceled = True
                    self.status_message = f"Analysis stopped after reading {frame_count} frames"
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
                frame_count += 1
                
                # Update progress every 100 frames
                if frame_count % 100 == 0:
                    self.progress = min(20, int((frame_count / total_frames) * 20))  # Max 20% for reading
                    self.status_message = f"Read {frame_count} frames..."
            
            total_frames = len(all_frames)
            self.status_message = f"Processing {total_frames} frames at {fps} FPS"
            self.progress = 20  # Reading complete - 20%

            allow_early_finish = not any(frame is not None for frame in self._key_frames.values())

            if analysis_canceled:
                if self.output_video_path and os.path.exists(self.output_video_path):
                    try:
                        os.remove(self.output_video_path)
                    except Exception:
                        pass

                self._sync_job_state(
                    status=AnalysisJob.STATUS_CANCELED,
                    message=self.status_message,
                    finished_at=timezone.now(),
                    cancel_requested=True
                )

                return {
                    'status': 'canceled',
                    'message': self.status_message
                }
            
            # Process frames with selective processing
            
            # Prepare for MP4 encoding using FFmpeg
            mp4_output = self.output_video_path
            
            # Build the FFmpeg command for MP4 encoding
            mp4_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-f', 'image2pipe',  # Input format is image pipe
                '-vcodec', 'png',  # Input codec is PNG (from OpenCV imencode)
                '-r', str(fps),  # Input frame rate
                '-i', '-',  # Input from stdin
                '-c:v', 'libx264',  # H.264 codec
                '-crf', '23',  # Constant Rate Factor (quality level - lower is better)
                '-preset', 'medium',  # Encoding speed/compression ratio
                '-profile:v', 'high444',  # H.264 profile that supports 4:4:4 color
                '-level', '4.0',  # H.264 level (increased for better compatibility)
                '-pix_fmt', 'yuv420p',  # Convert to a widely supported pixel format
                mp4_output
            ]
            
            # Start FFmpeg process
            self.status_message = "Starting MP4 encoding..."
            print(f"Running MP4 encoding with command: {' '.join(mp4_cmd)}")
            
            ffmpeg_process: Any = subprocess.Popen(
                mp4_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            print("Successfully started FFmpeg process for MP4 encoding")

            for i, frame in enumerate(all_frames):
                # Check if stop was requested
                if self._analysis_should_stop():
                    analysis_canceled = True
                    self.status_message = f"Analysis stopped after processing {i} frames"
                    break
                
                self._frame_count += 1
                
                # Update progress periodically
                if i % 10 == 0:
                    # Scale from 20% to 90% for processing
                    self.progress = 20 + int((i / total_frames) * 70)
                    self.status_message = f"Processing frame {i}/{total_frames} ({self.progress}% complete)"
                
                # Determine if this frame should be processed
                should_process = False
                
                # Check if any key frames are missing
                missing_key_frames = (
                        self.pre_hit_frame is None or
                        self.hit_frame is None or
                        self.follow_through_frame is None
                )
                
                # Process frame if it's on the interval, or we're still looking for key frames
                if self._frame_count % self._process_interval == 0 or missing_key_frames:
                    should_process = True
                
                if should_process:
                    # Calculate timestamp based on frame position and FPS
                    current_time = (i / fps) if fps > 0 else i * 0.033  # Fallback to 30fps if fps is invalid
                    processed_frame, analysis = self.process_frame(frame, current_time)
                    
                    if processed_frame is not None:
                        # Write processed frame to FFmpeg stdin
                        success, buffer = cv2.imencode('.png', processed_frame)
                        if success:
                            ffmpeg_process.stdin.write(self._buffer_to_bytes(buffer))
                        else:
                            print(f"Warning: Failed to encode processed frame {i} for FFmpeg")
                            # Fallback to writing original frame if encoding fails
                            success, buffer = cv2.imencode('.png', frame)
                            if success:
                                ffmpeg_process.stdin.write(self._buffer_to_bytes(buffer))
                            else:
                                print(f"Error: Failed to encode original frame {i} for FFmpeg")
                else:
                    # Write unprocessed frame to FFmpeg stdin
                    success, buffer = cv2.imencode('.png', frame)
                    if success:
                        ffmpeg_process.stdin.write(self._buffer_to_bytes(buffer))
                    else:
                        print(f"Error: Failed to encode unprocessed frame {i} for FFmpeg")
                
                # Break early if we have all key frames to save processing time
                if (allow_early_finish and
                    self.pre_hit_frame is not None and
                        self.hit_frame is not None and
                        self.follow_through_frame is not None):
                    self.status_message = f"All key frames detected after {i + 1}/{total_frames} frames. Finishing..."
                    
                    # Write remaining frames to the output
                    for remaining_frame in all_frames[i + 1:]:
                        success, buffer = cv2.imencode('.png', remaining_frame)
                        if success:
                            ffmpeg_process.stdin.write(self._buffer_to_bytes(buffer))
                        else:
                            print(f"Error: Failed to encode remaining frame for FFmpeg")
                    break

            if analysis_canceled:
                try:
                    if ffmpeg_process.stdin:
                        ffmpeg_process.stdin.close()
                    ffmpeg_process.terminate()
                except Exception:
                    pass

                if self.output_video_path and os.path.exists(self.output_video_path):
                    try:
                        os.remove(self.output_video_path)
                    except Exception:
                        pass

                self.progress = min(self.progress, 100)
                self._sync_job_state(
                    status=AnalysisJob.STATUS_CANCELED,
                    message=self.status_message,
                    finished_at=timezone.now(),
                    cancel_requested=True
                )

                return {
                    'status': 'canceled',
                    'message': self.status_message
                }
            
            # Close FFmpeg stdin to signal end of input
            ffmpeg_process.stdin.close()
            
            # Wait for FFmpeg process to finish
            stdout, stderr = ffmpeg_process.communicate()
            
            if ffmpeg_process.returncode != 0:
                print(f"FFmpeg encoding failed with error code {ffmpeg_process.returncode}")
                print(f"FFmpeg stderr:\n{stderr.decode()}")
                raise Exception("MP4 encoding failed")
            
            print("MP4 encoding completed successfully")
            
            # Generate final summary
            self.status_message = "Finalizing analysis..."
            self.progress = 90  # Processing complete - 90%
            
            # Helper function to make JSON serializable
            def make_json_serializable(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()  # Convert array to list
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj
            
            # Get technique summary from tracker
            raw_summary = self.tracker.get_technique_summary()

            # Process the summary to make it JSON serializable
            self.analysis_summary = make_json_serializable(raw_summary)

            # --- Generate AI Feedback using Ollama ---
            if self.analysis_summary.get('status') == 'complete':
                self.status_message = "Generating AI feedback..."
                self.progress = 95 # Update progress
                try:
                    ai_feedback = generate_feedback(self.analysis_summary)
                    self.analysis_summary['ai_feedback'] = ai_feedback
                    print("AI Feedback generated successfully.")
                except Exception as feedback_e:
                    print(f"Error generating AI feedback: {feedback_e}")
                    self.analysis_summary['ai_feedback'] = "Could not generate AI feedback due to an error."
            else:
                 self.analysis_summary['ai_feedback'] = "AI feedback skipped due to incomplete analysis."
            # --- End AI Feedback Generation ---

            # Save key frames to model
            self.save_key_frames()
            
            # Save output video to model
            if self.analysis_model and os.path.exists(self.output_video_path):
                with open(self.output_video_path, 'rb') as f:
                    file_content = f.read()
                    self.analysis_model.output_video.save(
                        os.path.basename(self.output_video_path),
                        ContentFile(file_content)
                    )
            
            # Save summary to model - summary is now JSON serializable
            if self.analysis_model:
                try:
                    # Final check to ensure JSON serialization works
                    json.dumps(self.analysis_summary)  # This will raise an error if not serializable
                    
                    self.analysis_model.analysis_summary = self.analysis_summary
                    self.analysis_model.is_analyzed = True
                    self.analysis_model.save()
                except TypeError as e:
                    print(f"Serialization error: {str(e)}")
                    # Create a simplified summary without the problematic data
                    self.analysis_model.analysis_summary = {
                        'status': raw_summary.get('status', 'error'),
                        'message': 'Analysis completed with serialization issues',
                        'shot_type': raw_summary.get('shot_type', '')
                    }
                    self.analysis_model.is_analyzed = True
                    self.analysis_model.save()
            
            self.status_message = "Analysis complete"
            self.progress = 100
            self._sync_job_state(
                status=AnalysisJob.STATUS_COMPLETED,
                message=self._status_message,
                progress=100,
                finished_at=timezone.now(),
                cancel_requested=False
            )
            
            return {
                'status': 'complete',
                'message': 'Video analysis completed successfully',
                'summary': self.analysis_summary
            }
        
        except Exception as e:
            self.status_message = f"Error during analysis: {str(e)}"
            self._sync_job_state(
                status=AnalysisJob.STATUS_FAILED,
                message=self._status_message,
                error_message=str(e),
                finished_at=timezone.now()
            )
            return {
                'status': 'error',
                'message': f"Analysis failed: {str(e)}"
            }
        
        finally:
            # Always ensure resources are released
            cap.release()
            # No OpenCV writer when using FFmpeg pipe
            self.cleanup()
            self._is_analyzing = False
    
    def cleanup(self) -> None:
        # Reset tracking state while preserving key frames
        self._start_time = None
        self._frame_count = 0
        self.stop_requested = False
