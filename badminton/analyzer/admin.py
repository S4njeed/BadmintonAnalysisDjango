"""Admin configuration for the Analyzer app."""

from django.contrib import admin
from .models import VideoAnalysis


@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    """Admin interface for VideoAnalysis model."""
    
    list_display = ('id', 'title', 'shot_type', 'created_at', 'is_analyzed')
    list_filter = ('shot_type', 'is_analyzed', 'created_at')
    search_fields = ('title',)
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'created_at', 'updated_at')
        }),
        ('Video Files', {
            'fields': ('input_video', 'output_video')
        }),
        ('Analysis Configuration', {
            'fields': ('shot_type', 'process_interval', 'source_fps')
        }),
        ('Analysis Results', {
            'fields': ('is_analyzed', 'analysis_summary', 'pre_hit_frame', 
                      'hit_frame', 'follow_through_frame')
        }),
    )
