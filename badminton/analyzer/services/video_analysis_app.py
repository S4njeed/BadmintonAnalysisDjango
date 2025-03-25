"""
VideoAnalysisApp - Adapted for Django integration with AV1 codec support.
Handles video processing and badminton technique analysis.
"""

import cv2
import time
import os
import json
import numpy as np
import subprocess
import tempfile
import shutil
from typing import Dict, Optional, Any, Tuple
from django.conf import settings
from django.core.files.base import ContentFile

from .badminton_pose_estimator import BadmintonVideoTracker


class DjangoVideoAnalysisApp:
    """Django-specific implementation of the video analysis application."""
    
    def __init__(self, analysis_model=None):
        """
        Initialize the analyzer with an optional analysis model instance.
        
        Args:
            analysis_model: Django model instance representing the analysis
        """
        # Django model for this analysis
        self.analysis_model = analysis_model
        
        # Core components
        self.tracker: Optional[BadmintonVideoTracker] = None
        self.current_video_path: Optional[str] = None
        self.analysis_summary: Optional[Dict[str, Any]] = None
        self.output_video_path: Optional[str] = None
        
        # Analysis state
        self._start_time: Optional[float] = None
        self._frame_count = 0
        self._is_analyzing = False
        self._stop_requested = False
        
        # Progress tracking for async feedback
        self._progress = 0
        self._status_message = "Initializing..."
        
        # Processing settings
        self._process_interval = 2  # Process every Nth frame
        self._source_fps = None  # Source video frame rate (None means auto-detect)
        
        # Key frames from shot analysis
        self._key_frames = {
            'pre_hit': None,
            'hit': None,
            'follow_through': None
        }
    
    @property
    def progress(self) -> int:
        """Return current progress percentage."""
        return self._progress
    
    @property
    def status_message(self) -> str:
        """Return current status message."""
        return self._status_message
    
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
        """Request analysis process to stop gracefully"""
        if not self._is_analyzing:
            return "No analysis is currently running"
        
        self._stop_requested = True
        return "Stop requested. Analysis will finish soon..."
    
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
            image_content = ContentFile(buffer.tobytes())
            
            # Save to the specified field
            field = getattr(self.analysis_model, field_name)
            field.save(f"{field_name}_{self.analysis_model.id}.jpg", image_content)
            
            return True
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
            return False
    
    def update_key_frames(self, new_frames: Dict[str, Optional[Dict]]) -> None:
        """Safely update stored key frames from tracker results"""
        # Check each phase (pre-hit, hit, follow-through)
        for phase in self._key_frames:
            # Only process if we have data for this phase
            if new_frames[phase] is not None:
                # Verify frame data exists before assignment
                if 'frame' in new_frames[phase] and isinstance(new_frames[phase]['frame'], np.ndarray):
                    self._key_frames[phase] = new_frames[phase]['frame'].copy()
    
    def initialize_tracker(self, shot_type: str) -> str:
        """Set up a new tracker instance for the specified shot type"""
        self.tracker = BadmintonVideoTracker(target_shot_type=shot_type)
        self._key_frames = {k: None for k in self._key_frames}
        self._frame_count = 0
        self._stop_requested = False
        return f"Initialized tracker for {shot_type.replace('_', ' ').title()} analysis"
    
    def process_frame(self, frame: np.ndarray, current_time: float) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Process a single video frame for pose analysis"""
        if frame is None:
            return None, None
        
        try:
            # Use frame skipping logic in our application layer
            if hasattr(self, 'last_processed_frame') and hasattr(self, 'last_analysis'):
                if self._frame_count % self._process_interval != 0:
                    # Return previous results for skipped frames
                    return self.last_processed_frame.copy(), self.last_analysis
            
            # Process the frame with the tracker
            annotated_frame, analysis = self.tracker.process_video_frame(
                frame,
                current_time,
                self._frame_count
            )
            
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
    
    def prepare_empty_frame(self) -> np.ndarray:
        """Create an empty frame with awaiting detection message."""
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
    
    def generate_frame_report(self, analysis: Optional[Dict]) -> str:
        """Generate analysis report for the current frame"""
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

    def create_hls_encoder(self, output_path, fps, frame_size):
        """
        Create an HLS encoder using FFmpeg.
        
        Args:
            output_path: Path to save the output video
            fps: Frames per second
            frame_size: Tuple of (width, height)
        
        Returns:
            Tuple of (temp_dir, frames_dir, is_available)
        """
        try:
            # Check if FFmpeg is installed
            try:
                subprocess.run(['ffmpeg', '-version'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              check=True)
                hls_available = True
                print("FFmpeg found and will be used for HLS encoding")
            except (subprocess.SubprocessError, FileNotFoundError):
                print("FFmpeg not found. Will fall back to OpenCV codecs.")
                return None, None, False
            
            # Create a temporary directory for frame extraction
            temp_dir = tempfile.mkdtemp(prefix="hls_encoding_")
            
            # Create a path for storing individual frames
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Store parameters for later use
            self._hls_frame_size = frame_size
            self._hls_fps = fps
            self._hls_frames_dir = frames_dir
            self._hls_frame_count = 0
            
            return temp_dir, frames_dir, True
        
        except Exception as e:
            print(f"Error initializing HLS encoder: {str(e)}")
            return None, None, False

    def save_frame_for_hls(self, frame, frame_dir, frame_index):
        """
        Save a single frame for later HLS encoding.
        
        Args:
            frame: NumPy array frame data
            frame_dir: Directory to save frames
            frame_index: Frame sequence number
            
        Returns:
            bool: True if saving was successful
        """
        try:
            # Format frame number with leading zeros
            frame_filename = os.path.join(frame_dir, f"frame_{frame_index:06d}.png")
            
            # Save using OpenCV's imwrite
            cv2.imwrite(frame_filename, frame)
            
            return True
        except Exception as e:
            print(f"Error saving frame for HLS: {str(e)}")
            return False

    def finalize_hls_encoding(self, temp_dir, frames_dir, output_path, frame_size, fps):
        """
        Perform the actual HLS encoding using FFmpeg.
        
        Args:
            temp_dir: Temporary directory
            frames_dir: Directory containing saved frames
            output_path: Base output path for the encoded video
            frame_size: Tuple of (width, height)
            fps: Frames per second
        
        Returns:
            bool: True if encoding was successful
        """
        try:
            width, height = frame_size
            frames_pattern = os.path.join(frames_dir, "frame_%06d.png")
            
            # Create HLS output directory
            hls_output_dir = os.path.join(settings.MEDIA_ROOT, 'hls')
            os.makedirs(hls_output_dir, exist_ok=True)
            
            # Get a unique identifier for this analysis stream
            analysis_id = "default"
            if self.analysis_model and self.analysis_model.id:
                analysis_id = str(self.analysis_model.id)
            
            # Create a unique directory for this analysis stream
            stream_dir = os.path.join(hls_output_dir, f"analysis_{analysis_id}")
            os.makedirs(stream_dir, exist_ok=True)
            
            # Define HLS output path
            hls_playlist = os.path.join(stream_dir, "playlist.m3u8")
            self.hls_playlist_path = hls_playlist
            
            # Store the relative path for Django reference
            rel_path = os.path.relpath(hls_playlist, settings.MEDIA_ROOT)
            # We need to store this path for later retrieval
            self.hls_relative_path = rel_path
            
            # Build the FFmpeg command for HLS encoding
            hls_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-framerate', str(fps),
                '-i', frames_pattern,
                '-c:v', 'libx264',  # H.264 codec for broader compatibility
                '-crf', '23',  # Constant Rate Factor (quality level - lower is better)
                '-preset', 'medium',  # Encoding speed/compression ratio
                '-profile:v', 'high444',  # H.264 profile that supports 4:4:4 color
                '-level', '4.0',  # H.264 level (increased for better compatibility)
                '-pix_fmt', 'yuv420p',  # Convert to a widely supported pixel format
                '-sc_threshold', '0',  # Disable scene change detection
                '-g', str(int(fps) * 2),  # GOP size (2 seconds)
                '-keyint_min', str(int(fps)),  # Minimum GOP size (1 second)
                '-hls_time', '4',  # Segment duration in seconds
                '-hls_playlist_type', 'vod',  # Video on demand playlist
                '-hls_segment_filename', os.path.join(stream_dir, "segment_%03d.ts"),
                '-hls_list_size', '0',  # Keep all segments in the playlist
                hls_playlist
            ]
            
            # Execute FFmpeg encoding
            self._status_message = "Performing HLS encoding..."
            print(f"Running HLS encoding with command: {' '.join(hls_cmd)}")
            
            result = subprocess.run(
                hls_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            print("HLS encoding completed successfully")
            
            # Also create a direct MP4 file for download purposes
            mp4_output = output_path  # This will be the normal .mp4 file path
            
            mp4_cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', frames_pattern,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-profile:v', 'high444',
                '-level', '4.0',
                '-pix_fmt', 'yuv420p',
                mp4_output
            ]
            
            print(f"Creating MP4 file for download with command: {' '.join(mp4_cmd)}")
            
            mp4_result = subprocess.run(
                mp4_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            print("MP4 creation completed successfully")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"HLS encoding failed: {e}")
            print(f"Error output: {e.stderr.decode()}")
            return False
        
        except Exception as e:
            print(f"Error during HLS encoding: {str(e)}")
            return False
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary files: {str(cleanup_error)}")
    
    def analyze_video(self) -> Dict[str, Any]:
        """
        Main method to analyze a video file.
        This is adapted from analyze_frame_by_frame to work in Django.
        """
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
        self._status_message = "Starting analysis..."
        self._progress = 0
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Larger buffer for smoother reading
        
        if not cap.isOpened():
            self._is_analyzing = False
            self._status_message = "Error opening video file"
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
        
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output directory and video writer
        media_root = settings.MEDIA_ROOT
        output_dir = os.path.join(media_root, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        output_filename = f"analysis_{timestamp}.mp4"  # Use .mp4 for direct download
        self.output_video_path = os.path.join(output_dir, output_filename)
        
        # Update status message
        self._status_message = f"Processing video ({total_frames} frames) at {fps:.1f} FPS, analyzing every {self._process_interval} frame"
        
        # Try to setup HLS encoder
        temp_dir, frames_dir, using_hls = self.create_hls_encoder(
            self.output_video_path,
            fps,
            frame_size
        )
        
        # If HLS setup fails, fall back to H.264 with OpenCV
        writer = None
        
        if not using_hls:
            print("Not using HLS, falling back to H.264 with OpenCV")
            # Use H.264 as only fallback - simpler than before
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
                writer = cv2.VideoWriter(
                    self.output_video_path,
                    fourcc,
                    fps,
                    frame_size
                )
                
                if not writer.isOpened():
                    raise Exception("Failed to initialize H.264 writer")
                
                print("Successfully opened H.264 writer")
            except Exception as e:
                print(f"H.264 setup failed: {str(e)}")
                raise Exception("Failed to initialize any video writer")
        
        self._start_time = time.time()
        
        try:
            # Read all frames at once for more efficient processing
            self._status_message = "Reading video frames..."
            all_frames = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
                frame_count += 1
                
                # Update progress every 100 frames
                if frame_count % 100 == 0:
                    self._progress = min(20, int((frame_count / total_frames) * 20))  # Max 20% for reading
                    self._status_message = f"Read {frame_count} frames..."
            
            total_frames = len(all_frames)
            self._status_message = f"Processing {total_frames} frames at {fps} FPS"
            self._progress = 20  # Reading complete - 20%
            
            # Process frames with selective processing
            self._hls_frame_count = 0  # Initialize frame counter for HLS
            
            for i, frame in enumerate(all_frames):
                # Check if stop was requested
                if self._stop_requested:
                    self._status_message = f"Analysis stopped after processing {i} frames"
                    break
                
                self._frame_count += 1
                
                # Update progress periodically
                if i % 10 == 0:
                    # Scale from 20% to 90% for processing
                    self._progress = 20 + int((i / total_frames) * 70)
                    remaining = total_frames - i
                    self._status_message = f"Processing frame {i}/{total_frames} ({self._progress}% complete)"
                
                # Determine if this frame should be processed
                should_process = False
                
                # Check if any key frames are missing
                missing_key_frames = (
                        self.pre_hit_frame is None or
                        self.hit_frame is None or
                        self.follow_through_frame is None
                )
                
                # Process frame if it's on the interval or we're still looking for key frames
                if self._frame_count % self._process_interval == 0 or missing_key_frames:
                    should_process = True
                
                if should_process:
                    # Calculate timestamp based on frame position and FPS
                    current_time = (i / fps) if fps > 0 else i * 0.033  # Fallback to 30fps if fps is invalid
                    processed_frame, analysis = self.process_frame(frame, current_time)
                    
                    if processed_frame is not None:
                        if using_hls:
                            # Save frames for HLS encoding
                            self.save_frame_for_hls(processed_frame, frames_dir, self._hls_frame_count)
                            self._hls_frame_count += 1
                        else:
                            # Write to OpenCV writer
                            writer.write(processed_frame)
                else:
                    # Write unprocessed frame
                    if using_hls:
                        # Save unprocessed frame for HLS encoding
                        self.save_frame_for_hls(frame, frames_dir, self._hls_frame_count)
                        self._hls_frame_count += 1
                    else:
                        # Write to OpenCV writer
                        writer.write(frame)
                
                # Break early if we have all key frames to save processing time
                if (self.pre_hit_frame is not None and
                        self.hit_frame is not None and
                        self.follow_through_frame is not None):
                    self._status_message = f"All key frames detected after {i + 1}/{total_frames} frames. Finishing..."
                    
                    # Write remaining frames to the output
                    for remaining_frame in all_frames[i + 1:]:
                        if using_hls:
                            # Save frames for HLS encoding
                            self.save_frame_for_hls(remaining_frame, frames_dir, self._hls_frame_count)
                            self._hls_frame_count += 1
                        else:
                            # Write to OpenCV writer
                            writer.write(remaining_frame)
                    break
            
            # Finalize video writing
            if using_hls:
                # Perform HLS encoding on collected frames
                self._status_message = "Performing HLS encoding..."
                success = self.finalize_hls_encoding(
                    temp_dir,
                    frames_dir,
                    self.output_video_path,
                    self._hls_frame_size,
                    self._hls_fps
                )
                
                if not success:
                    # If HLS encoding failed, we need to abort
                    raise Exception("HLS encoding failed, unable to produce output video")
            else:
                # Release OpenCV writer
                if writer:
                    writer.release()
            
            # Generate final summary
            self._status_message = "Finalizing analysis..."
            self._progress = 90  # Processing complete - 90%
            
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
            
            # Save HLS path to model if HLS was used
            if using_hls and self.analysis_model and hasattr(self, 'hls_relative_path'):
                self.analysis_model.hls_path = self.hls_relative_path
            
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
            
            self._status_message = "Analysis complete"
            self._progress = 100
            
            return {
                'status': 'complete',
                'message': 'Video analysis completed successfully',
                'summary': self.analysis_summary
            }
        
        except Exception as e:
            self._status_message = f"Error during analysis: {str(e)}"
            return {
                'status': 'error',
                'message': f"Analysis failed: {str(e)}"
            }
        
        finally:
            # Always ensure resources are released
            cap.release()
            if not using_hls and writer:
                writer.release()
            self.cleanup()
            self._is_analyzing = False
    
    def cleanup(self) -> None:
        """Reset tracking state while preserving key frames"""
        self._start_time = None
        self._frame_count = 0
        self._stop_requested = False