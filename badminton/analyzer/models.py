"""Models for the Analyzer app."""

import os
import uuid
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _


def video_upload_path(instance, filename):
    """Generate a unique path for uploaded videos."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('videos', filename)


def output_upload_path(instance, filename):
    """Generate a unique path for analyzed videos."""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('outputs', filename)


class VideoAnalysis(models.Model):
    """Model to store video analysis data."""
    
    # Shot type choices
    SHOT_TYPES = [
        ('smash', 'Smash'),
        ('clear', 'Clear'),
        ('drop_shot', 'Drop Shot'),
    ]
    
    # Basic metadata
    title = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Video files
    input_video = models.FileField(
        upload_to=video_upload_path,
        verbose_name=_('Input Video'),
    )
    output_video = models.FileField(
        upload_to=output_upload_path,
        verbose_name=_('Analyzed Video'),
        blank=True,
        null=True,
    )
    hls_path = models.CharField(
        max_length=255,
        verbose_name=_('HLS Playlist Path'),
        blank=True,
        null=True,
        help_text=_('Path to the HLS playlist file for streaming')
    )
    
    # Analysis configuration
    shot_type = models.CharField(
        max_length=20,
        choices=SHOT_TYPES,
        default='smash',
        verbose_name=_('Shot Type'),
    )
    process_interval = models.PositiveIntegerField(
        default=2,
        verbose_name=_('Frame Skip Interval'),
        help_text=_('Process every Nth frame (1=all frames, higher=faster)')
    )
    source_fps = models.FloatField(
        blank=True,
        null=True,
        verbose_name=_('Source Video FPS'),
        help_text=_('Leave empty for auto-detection')
    )
    
    # Analysis results
    is_analyzed = models.BooleanField(default=False)
    analysis_summary = models.JSONField(blank=True, null=True)
    
    # Frame images
    pre_hit_frame = models.ImageField(
        upload_to='frames',
        blank=True,
        null=True,
        verbose_name=_('Pre-Hit Frame')
    )
    hit_frame = models.ImageField(
        upload_to='frames',
        blank=True,
        null=True,
        verbose_name=_('Hit Frame')
    )
    follow_through_frame = models.ImageField(
        upload_to='frames',
        blank=True,
        null=True,
        verbose_name=_('Follow-Through Frame')
    )
    
    class Meta:
        verbose_name = _('Video Analysis')
        verbose_name_plural = _('Video Analyses')
        ordering = ['-created_at']
    
    def __str__(self):
        """String representation of the model."""
        if self.title:
            return self.title
        return f"Analysis {self.id} - {self.shot_type}"
    
    def delete(self, *args, **kwargs):
        """Override delete to remove files."""
        # Delete files from storage
        if self.input_video:
            self.input_video.delete(save=False)
        if self.output_video:
            self.output_video.delete(save=False)
        if self.pre_hit_frame:
            self.pre_hit_frame.delete(save=False)
        if self.hit_frame:
            self.hit_frame.delete(save=False)
        if self.follow_through_frame:
            self.follow_through_frame.delete(save=False)
            
        # Clean up HLS files if they exist
        if self.hls_path:
            try:
                hls_dir = os.path.dirname(os.path.join(settings.MEDIA_ROOT, self.hls_path))
                if os.path.exists(hls_dir):
                    import shutil
                    shutil.rmtree(hls_dir)
            except Exception as e:
                print(f"Error removing HLS directory: {str(e)}")
            
        # Call the parent's delete method
        super().delete(*args, **kwargs)
