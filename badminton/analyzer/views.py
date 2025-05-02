"""
Views for the Analyzer app - handles HTTP requests and renders responses.
"""

import os
import json
import threading
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.urls import reverse
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required # Import login_required
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.contrib.auth import login # Import login function
from django.urls import reverse_lazy # For redirecting after signup

from .models import VideoAnalysis
from .forms import VideoUploadForm, SignUpForm # Import SignUpForm
from .services.video_analysis_app import DjangoVideoAnalysisApp


# Global dictionary to track running analysis tasks
running_analyses = {}


def home(request):
    """Home page view."""
    form = VideoUploadForm()
    context = {
        'form': form,
        'recent_analyses': VideoAnalysis.objects.filter(is_analyzed=True).order_by('-created_at')[:3],
    }
    return render(request, 'analyzer/index.html', context)

@login_required # Require login for upload
def upload_video(request):
    """Handle video upload and initial analysis configuration."""
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Don't save immediately, set the user first
            analysis = form.save(commit=False)
            analysis.user = request.user # Associate the logged-in user
            analysis.save() # Now save the instance
            return redirect('analyzer:analysis_detail', analysis_id=analysis.id)
    else:
        form = VideoUploadForm()
    
    return render(request, 'analyzer/upload.html', {'form': form})


def analysis_detail(request, analysis_id):
    """Display detailed view of an analysis."""
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    
    # Convert to string for consistent lookup
    str_analysis_id = str(analysis_id)
    
    context = {
        'analysis': analysis,
        'is_running': str_analysis_id in running_analyses,
        'MEDIA_URL': settings.MEDIA_URL,  # Add MEDIA_URL for HLS streaming paths
    }
    return render(request, 'analyzer/analysis_detail.html', context)

@login_required # Require login to view history
def analysis_history(request):
    """Display user's analysis history."""
    # Filter analyses by the currently logged-in user
    analyses = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'analyzer/history.html', {'analyses': analyses})


@csrf_exempt
def start_analysis(request, analysis_id):
    """Start the analysis process for a video."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)
    
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    
    # Don't restart if already running
    if analysis_id in running_analyses:
        return JsonResponse({
            'status': 'running',
            'message': 'Analysis already running'
        })
    
    # Clear previous results if any
    if analysis.is_analyzed:
        analysis.is_analyzed = False
        analysis.output_video = None
        analysis.pre_hit_frame = None
        analysis.hit_frame = None
        analysis.follow_through_frame = None
        analysis.analysis_summary = None
        analysis.save()
    
    # Create the analyzer instance
    analyzer = DjangoVideoAnalysisApp(analysis)
    
    # Store in running analyses - ensure the key is a string
    str_analysis_id = str(analysis_id)
    running_analyses[str_analysis_id] = {
        'analyzer': analyzer,
        'progress': 0,
        'status': 'starting',
        'message': 'Initializing analysis...'
    }
    
    # Start analysis in a background thread
    thread = threading.Thread(
        target=run_analysis_thread,
        args=(analysis_id, analyzer)
    )
    thread.daemon = True
    thread.start()
    
    return JsonResponse({
        'status': 'started',
        'message': 'Analysis started successfully'
    })


def run_analysis_thread(analysis_id, analyzer):
    """Run the analysis in a background thread."""
    # Convert to string for consistent lookup
    str_analysis_id = str(analysis_id)
    
    try:
        print(f"Starting analysis thread for ID: {str_analysis_id}")
        running_analyses[str_analysis_id]['status'] = 'running'
        
        # Run the analysis
        result = analyzer.analyze_video()
        
        # Update the status
        if result.get('status') == 'complete':
            running_analyses[str_analysis_id]['status'] = 'completed'
            running_analyses[str_analysis_id]['message'] = 'Analysis completed successfully'
            
            # Important: Clean up running_analyses after successful completion
            # Wait 10 seconds to ensure client has time to check progress one last time
            import time
            time.sleep(10)
            if str_analysis_id in running_analyses:
                del running_analyses[str_analysis_id]
        else:
            running_analyses[str_analysis_id]['status'] = 'error'
            running_analyses[str_analysis_id]['message'] = result.get('message', 'Analysis failed')
            
    except Exception as e:
        # Handle exceptions
        print(f"Error in analysis thread: {str(e)}")
        running_analyses[str_analysis_id]['status'] = 'error'
        running_analyses[str_analysis_id]['message'] = str(e)
    
    # Keep the result in running_analyses for a short time so the client can fetch it
    # A proper cleanup mechanism would be needed in a production system


@csrf_exempt
def analysis_progress(request, analysis_id):
    """Get the current progress of an analysis."""
    try:
        # Check if the analysis exists and get the model
        analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
        
        # Convert analysis_id to a string to be safe
        analysis_id_str = str(analysis_id)
        
        # Check if the analysis is running
        if analysis_id_str in running_analyses:
            # Get the analyzer data from running analyses
            analyzer_data = running_analyses[analysis_id_str]
            
            # Get the analyzer instance
            if 'analyzer' in analyzer_data:
                analyzer = analyzer_data['analyzer']
                
                # Debug info about the analyzer state
                print(f"Analysis {analysis_id} status: {analyzer_data['status']}")
                print(f"Progress: {analyzer.progress}%, Message: {analyzer.status_message}")
                
                # Return the progress info
                return JsonResponse({
                    'status': analyzer_data['status'],
                    'progress': analyzer.progress,
                    'message': analyzer.status_message
                })
            else:
                # No analyzer in the data
                return JsonResponse({
                    'status': 'error',
                    'progress': 0,
                    'message': 'Analysis configuration error'
                })
        
        # Not in running_analyses, check if it's completed
        if analysis.is_analyzed:
            return JsonResponse({
                'status': 'completed',
                'progress': 100,
                'message': 'Analysis completed',
                'result_url': reverse('analyzer:analysis_detail', args=[analysis_id])
            })
        else:
            return JsonResponse({
                'status': 'not_started',
                'progress': 0,
                'message': 'Analysis has not been started'
            })
    except Exception as e:
        # Log any unexpected errors
        print(f"Error in analysis_progress: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'progress': 0,
            'message': f'Error checking progress: {str(e)}'
        })


@csrf_exempt
def stop_analysis(request, analysis_id):
    """Stop a running analysis."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST required'}, status=405)
    
    # Convert to string for consistent lookup
    str_analysis_id = str(analysis_id)
    
    if str_analysis_id not in running_analyses:
        return JsonResponse({
            'status': 'error',
            'message': 'No analysis running with this ID'
        })
    
    # Request the analyzer to stop
    analyzer = running_analyses[str_analysis_id]['analyzer']
    response = analyzer.request_stop()
    
    return JsonResponse({
        'status': 'stopping',
        'message': response
    })


def download_analysis(request, analysis_id):
    """Download the analyzed video."""
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    
    if not analysis.output_video:
        raise Http404("No analyzed video available")
    
    # Determine content type based on file extension
    file_path = analysis.output_video.path
    file_name = os.path.basename(file_path)
    
    # Set appropriate content type
    content_type = 'video/mp4'  # Default
    if file_name.endswith('.avi'):
        content_type = 'video/x-msvideo'
    elif file_name.endswith('.mov'):
        content_type = 'video/quicktime'
    elif file_name.endswith('.webm'):
        content_type = 'video/webm'
    
    # Serve the file
    response = HttpResponse(analysis.output_video.read(), content_type=content_type)
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    return response


@login_required
def delete_analysis(request, analysis_id):
    """Delete a specific analysis entry belonging to the user."""
    # Ensure it's a POST request for safety
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
    return redirect('analyzer:analysis_history')


def signup(request):
    """Handle user registration."""
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user) # Log the user in automatically after signup
            messages.success(request, "Registration successful. You are now logged in.")
            return redirect('analyzer:home') # Redirect to home page after signup
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})

    # Get the analysis object, ensuring it belongs to the current user
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id, user=request.user)

    try:
        analysis_title = analysis.title or f"Analysis {analysis.id}"
        analysis.delete() # This calls the overridden delete method in the model
        messages.success(request, f'Successfully deleted "{analysis_title}".')
    except Exception as e:
        messages.error(request, f"Error deleting analysis: {str(e)}")

    return redirect('analyzer:analysis_history')
