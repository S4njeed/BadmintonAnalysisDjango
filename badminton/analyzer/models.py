"""Models for the Analyzer app."""

import os
import uuid
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User  # Import the User model
from django.utils import timezone
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
    
    # User association
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,  # Delete analysis if user is deleted
        related_name='video_analyses',
        verbose_name=_('User'),
        null=True,  # Allow null for now, maybe handle anonymous uploads later
        blank=True, # Allow blank in forms/admin
    )

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
        return f"Analysis {self.pk} - {self.shot_type}"
    
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
            
        # Call the parent's delete method
        super().delete(*args, **kwargs)


class AnalysisJob(models.Model):
    """Persistent job state for a video analysis run."""

    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELED = "canceled"

    STATUS_CHOICES = [
        (STATUS_QUEUED, _('Queued')),
        (STATUS_RUNNING, _('Running')),
        (STATUS_COMPLETED, _('Completed')),
        (STATUS_FAILED, _('Failed')),
        (STATUS_CANCELED, _('Canceled')),
    ]

    analysis = models.OneToOneField(
        VideoAnalysis,
        on_delete=models.CASCADE,
        related_name='job',
        verbose_name=_('Video Analysis'),
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_QUEUED,
        verbose_name=_('Status'),
    )
    progress = models.PositiveSmallIntegerField(default=0, verbose_name=_('Progress'))
    message = models.TextField(blank=True, default='', verbose_name=_('Status Message'))
    error_message = models.TextField(blank=True, default='', verbose_name=_('Error Message'))
    cancel_requested = models.BooleanField(default=False, verbose_name=_('Cancel Requested'))
    started_at = models.DateTimeField(blank=True, null=True, verbose_name=_('Started At'))
    finished_at = models.DateTimeField(blank=True, null=True, verbose_name=_('Finished At'))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('Analysis Job')
        verbose_name_plural = _('Analysis Jobs')
        ordering = ['-created_at']

    def __str__(self):
        analysis_pk = getattr(self.analysis, 'pk', 'unknown')
        return f"Analysis Job {self.pk} - Analysis {analysis_pk} ({self.status})"

    def reset_for_run(self, message='Queued for analysis'):
        self.status = self.STATUS_QUEUED
        self.progress = 0
        self.message = message
        self.error_message = ''
        self.cancel_requested = False
        self.started_at = None
        self.finished_at = None
        self.save()

    def mark_running(self, message='Analysis in progress'):
        self.status = self.STATUS_RUNNING
        self.message = message
        if self.started_at is None:
            self.started_at = timezone.now()
        self.finished_at = None
        self.save()

    def mark_completed(self, message='Analysis completed successfully'):
        self.status = self.STATUS_COMPLETED
        self.progress = 100
        self.message = message
        self.error_message = ''
        self.cancel_requested = False
        self.finished_at = timezone.now()
        self.save()

    def mark_failed(self, message='Analysis failed', error_message=''):
        self.status = self.STATUS_FAILED
        self.message = message
        self.error_message = error_message or message
        self.cancel_requested = False
        self.finished_at = timezone.now()
        self.save()

    def mark_canceled(self, message='Analysis canceled'):
        self.status = self.STATUS_CANCELED
        self.message = message
        self.cancel_requested = True
        self.finished_at = timezone.now()
        self.save()

    @property
    def is_active(self):
        return self.status in {self.STATUS_QUEUED, self.STATUS_RUNNING}
