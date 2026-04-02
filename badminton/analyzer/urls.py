"""URL patterns for the Analyzer app."""

from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('upload/', views.upload_video, name='upload_video'),
    path('analysis/<int:analysis_id>/', views.analysis_detail, name='analysis_detail'),
    path('history/', views.analysis_history, name='analysis_history'),
    # Removed signup URL from here

    # AJAX endpoints
    path('start-analysis/<int:analysis_id>/', views.start_analysis, name='start_analysis'),
    path('analysis-progress/<int:analysis_id>/', views.analysis_progress, name='analysis_progress'),
    path('stop-analysis/<int:analysis_id>/', views.stop_analysis, name='stop_analysis'),
    
    # Download endpoints
    path('download/<int:analysis_id>/', views.download_analysis, name='download_analysis'),
    path('download-images/<int:analysis_id>/', views.download_analysis_images, name='download_analysis_images'),

    # Action endpoints
    path('delete/<int:analysis_id>/', views.delete_analysis, name='delete_analysis'),
]
