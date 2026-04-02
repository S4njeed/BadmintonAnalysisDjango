"""
Views for the Analyzer app - handles HTTP requests and renders responses.
"""

import os
import threading
from io import BytesIO
from mimetypes import guess_type
from zipfile import ZIP_DEFLATED, ZipFile

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login  # Import login function
from django.contrib.auth.decorators import login_required  # Import login_required
from django.http import FileResponse, HttpResponse, Http404, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt

from .forms import VideoUploadForm, SignUpForm  # Import SignUpForm
from .models import VideoAnalysis, AnalysisJob
from .services.video_analysis_app import DjangoVideoAnalysisApp


ACTIVE_JOB_STATUSES = {
    AnalysisJob.STATUS_QUEUED,
    AnalysisJob.STATUS_RUNNING,
}


def _clear_previous_analysis_outputs(analysis):
    """Remove any previously generated files before rerunning analysis."""
    for field_name in ('output_video', 'pre_hit_frame', 'hit_frame', 'follow_through_frame'):
        field = getattr(analysis, field_name)
        if field:
            field.delete(save=False)
            setattr(analysis, field_name, None)

    analysis.analysis_summary = None
    analysis.is_analyzed = False
    analysis.save(update_fields=['output_video', 'pre_hit_frame', 'hit_frame', 'follow_through_frame', 'analysis_summary', 'is_analyzed', 'updated_at'])


def _get_analysis_job(analysis):
    return AnalysisJob.objects.filter(analysis=analysis).first()


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
    analysis_job = _get_analysis_job(analysis)
    
    context = {
        'analysis': analysis,
        'analysis_job': analysis_job,
        'is_running': bool(analysis_job and analysis_job.status in ACTIVE_JOB_STATUSES),
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
    job, created = AnalysisJob.objects.get_or_create(analysis=analysis)
    
    # Don't restart if already running or queued
    if not created and job.status in ACTIVE_JOB_STATUSES:
        return JsonResponse({
            'status': 'running',
            'progress': job.progress,
            'message': job.message or 'Analysis already running'
        })
    
    # Clear previous results if any before rerunning
    if analysis.is_analyzed or analysis.output_video or analysis.pre_hit_frame or analysis.hit_frame or analysis.follow_through_frame:
        _clear_previous_analysis_outputs(analysis)
    
    # Create the analyzer instance
    analyzer = DjangoVideoAnalysisApp(analysis, analysis_job=job)

    job.reset_for_run('Initializing analysis...')
    
    # Start analysis in a background thread
    thread = threading.Thread(
        target=run_analysis_thread,
        args=(analysis_id, analyzer, job.id)
    )
    thread.daemon = True
    thread.start()
    
    return JsonResponse({
        'status': 'started',
        'progress': job.progress,
        'message': 'Analysis started successfully'
    })


def run_analysis_thread(analysis_id, analyzer, job_id):
    """Run the analysis in a background thread."""
    try:
        job = AnalysisJob.objects.get(id=job_id)
        if job.cancel_requested:
            job.mark_canceled('Analysis canceled before start')
            return
        job.mark_running('Running analysis...')
        
        # Run the analysis
        result = analyzer.analyze_video()
        
        # Update the status
        if result.get('status') == 'complete':
            job.mark_completed(result.get('message', 'Analysis completed successfully'))
        elif result.get('status') == 'canceled':
            job.mark_canceled(result.get('message', 'Analysis canceled'))
        else:
            job.mark_failed(
                result.get('message', 'Analysis failed'),
                result.get('message', 'Analysis failed')
            )
            
    except Exception as e:
        # Handle exceptions
        print(f"Error in analysis thread: {str(e)}")
        try:
            job = AnalysisJob.objects.get(id=job_id)
            job.mark_failed('Analysis failed', str(e))
        except AnalysisJob.DoesNotExist:
            pass


@csrf_exempt
def analysis_progress(request, analysis_id):
    """Get the current progress of an analysis."""
    try:
        analysis = get_object_or_404(VideoAnalysis, id=analysis_id)

        job = _get_analysis_job(analysis)
        if job:
            if job.status in ACTIVE_JOB_STATUSES:
                if job.cancel_requested:
                    return JsonResponse({
                        'status': 'canceling',
                        'progress': job.progress,
                        'message': job.message or 'Cancellation requested. Stopping analysis...'
                    })

                return JsonResponse({
                    'status': job.status,
                    'progress': job.progress,
                    'message': job.message or 'Processing analysis...'
                })

            if job.status == AnalysisJob.STATUS_COMPLETED and analysis.is_analyzed:
                return JsonResponse({
                    'status': 'completed',
                    'progress': 100,
                    'message': job.message or 'Analysis completed',
                    'result_url': reverse('analyzer:analysis_detail', args=[analysis_id])
                })

            if job.status == AnalysisJob.STATUS_CANCELED:
                return JsonResponse({
                    'status': 'canceled',
                    'progress': job.progress,
                    'message': job.message or 'Analysis canceled'
                })

            if job.status == AnalysisJob.STATUS_FAILED:
                return JsonResponse({
                    'status': 'error',
                    'progress': job.progress,
                    'message': job.error_message or job.message or 'Analysis failed'
                })

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
    
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    job = _get_analysis_job(analysis)

    if job is None or job.status not in ACTIVE_JOB_STATUSES:
        return JsonResponse({
            'status': 'error',
            'message': 'No analysis running with this ID'
        })
    
    job.cancel_requested = True
    job.message = 'Cancellation requested. Stopping analysis...'
    job.save(update_fields=['cancel_requested', 'message', 'updated_at'])
    
    return JsonResponse({
        'status': 'canceling',
        'message': 'Cancellation requested. Stopping analysis...'
    })


def download_analysis(request, analysis_id):
    """Download the analyzed video."""
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)
    
    if not analysis.output_video:
        raise Http404("No analyzed video available")
    
    stored_file_name = os.path.basename(analysis.output_video.name or analysis.output_video.path)
    download_extension = os.path.splitext(stored_file_name)[1] or '.mp4'
    download_name = f'analysis_{analysis.id}_video{download_extension}'
    content_type = guess_type(stored_file_name)[0] or 'video/mp4'

    try:
        analysis.output_video.open('rb')
    except FileNotFoundError:
        raise Http404("Analyzed video file is missing")

    return FileResponse(
        analysis.output_video,
        as_attachment=True,
        filename=download_name,
        content_type=content_type,
    )


def download_analysis_images(request, analysis_id):
    """Download the analyzed key frames as a ZIP archive."""
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id)

    frame_fields = [
        ('pre_hit', analysis.pre_hit_frame),
        ('hit', analysis.hit_frame),
        ('follow_through', analysis.follow_through_frame),
    ]

    archive_buffer = BytesIO()
    files_added = 0

    with ZipFile(archive_buffer, 'w', ZIP_DEFLATED) as zip_file:
        for phase, frame_field in frame_fields:
            if not frame_field:
                continue

            try:
                frame_field.open('rb')
                zip_file.writestr(
                    f'analysis_{analysis.id}_{phase}.jpg',
                    frame_field.read(),
                )
                files_added += 1
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error adding {phase} frame to archive: {e}")
            finally:
                try:
                    frame_field.close()
                except Exception:
                    pass

    if files_added == 0:
        raise Http404("No analyzed images available")

    archive_buffer.seek(0)
    response = HttpResponse(archive_buffer.getvalue(), content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="analysis_{analysis.id}_frames.zip"'
    return response


@login_required
def delete_analysis(request, analysis_id):
    """Delete a specific analysis entry belonging to the user."""
    # Ensure it's a POST request for safety
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect('analyzer:analysis_history')

    # Get the analysis object, ensuring it belongs to the current user
    analysis = get_object_or_404(VideoAnalysis, id=analysis_id, user=request.user)

    try:
        analysis_title = analysis.title or f"Analysis {analysis.id}"
        analysis.delete()  # Delete the analysis object
        messages.success(request, f'Successfully deleted "{analysis_title}".')
    except Exception as e:
        messages.error(request, f"Error deleting analysis: {str(e)}")

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
