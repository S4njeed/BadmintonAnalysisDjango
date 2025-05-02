# Badminton Technique Analyzer - Implementation Documentation

## 1. Project Aim and Overview

This document details the implementation of the "Badminton Pose Estimation and Feedback System," a final year project aimed at addressing the need for accessible badminton technique analysis. The system leverages computer vision (specifically pose estimation and object detection) to analyze user-uploaded videos of badminton shots.

**Core Objectives:**
-   Develop a system to process uploaded badminton videos.
-   Implement pose estimation (using YOLOv11) to track player keypoints.
-   Implement shuttlecock detection (using a custom YOLO model) to aid analysis.
-   Identify key phases of a badminton shot (pre-hit, hit, follow-through) based on biomechanics and shuttlecock proximity.
-   Calculate relevant metrics (e.g., grip angle, shoulder angle).
-   Provide visual feedback through annotated video frames and key frame extraction.
-   (Future Goal) Generate corrective textual feedback using LLMs.

The project followed an Agile methodology, iterating through prototypes and incorporating refinements. A key architectural decision was the planned migration from an initial **Gradio-based prototype** to a more robust **Django web application**. Gradio was instrumental for rapid prototyping and validating the core analysis logic, while Django was chosen for building a scalable, feature-rich application with user management and persistent data storage. This document details the final planned Django architecture and the underlying analysis engine, referencing the Gradio origins where relevant to illustrate the evolution.

## 2. System Architecture and Technology Stack (Django Implementation)

The target architecture is a full-stack web application built on Django, designed to handle user interactions, manage data, and orchestrate backend analysis tasks.

-   **Initial Prototyping Tool:** Gradio. Used for early versions (V1-V3) to quickly build an interactive interface around the Python analysis functions, allowing for rapid testing of the pose estimation and feedback concepts without extensive web development overhead.
-   **Target Backend Framework:** Django. Selected for the full application due to its Model-View-Template (MVT) architecture, built-in Object-Relational Mapper (ORM) for database interaction, user authentication system, admin interface, form handling, and overall scalability compared to Gradio.
-   **Target Frontend:** Standard web technologies (HTML, CSS, JavaScript/TypeScript) rendered via Django Templates. Bootstrap CSS was used for initial styling. While Gradio automatically generates a UI, Django requires explicit frontend development, offering greater control and customization.
-   **Core Processing Language:** Python 3.x. Consistent choice across both Gradio and Django phases due to its strong ecosystem for CV/ML tasks.
-   **Computer Vision Library:** OpenCV (`cv2`). Used for video reading (frame extraction), image manipulation, and drawing visualizations (skeletons, bounding boxes, text).
-   **Machine Learning Framework:** PyTorch. Used as the underlying framework for running the YOLO models.
-   **Pose Estimation Model:** YOLOv11-Pose (`yolo11n-pose.pt`). Selected after consideration of alternatives like MediaPipe, aiming for a balance of accuracy and performance. The 'n' (nano) version was used for efficiency. GPU acceleration (`cuda`) is utilized if available.
-   **Object Detection Model:** Custom-trained YOLO model (`shuttlecock.pt`) via Roboflow. Specifically trained to detect badminton shuttlecocks.
-   **ML Model Interface:** Ultralytics library. Used to load and run inference with the YOLO models.
-   **Video Encoding:** FFmpeg. Used via `subprocess` to encode annotated frames into a standard MP4 video format suitable for web playback, offering more control and compatibility than OpenCV's `VideoWriter` alone.
-   **Database:** SQLite (Default for Django development). Used to store user information and `VideoAnalysis` records.
-   **Development Environment:** VS Code / PyCharm, Anaconda (managing Python environments), Git (version control).

## 3. Project Structure (Planned Django Implementation)

The planned Django structure organizes the project into configuration and application-specific modules.

```
.
├── .gitignore                     # Git ignore file
├── badminton/                     # Root Django Project Directory
│   ├── badminton/                 # Project Configuration Package
│   │   ├── __init__.py
│   │   ├── asgi.py                # ASGI entry point
│   │   ├── settings.py            # Project settings
│   │   ├── urls.py                # Root URL configuration
│   │   └── wsgi.py                # WSGI entry point
│   ├── analyzer/                  # Core 'analyzer' Django Application
│   │   ├── __init__.py
│   │   ├── admin.py               # Admin interface config
│   │   ├── apps.py                # App configuration
│   │   ├── forms.py               # Django Forms (VideoUploadForm, SignUpForm)
│   │   ├── models.py              # Database Models (VideoAnalysis)
│   │   ├── tests.py               # Unit tests
│   │   ├── urls.py                # App-level URLs
│   │   ├── views.py               # Request handling logic
│   │   ├── migrations/            # Database migrations
│   │   ├── services/              # Core analysis logic
│   │   │   ├── __init__.py
│   │   │   ├── badminton_pose_estimator.py # Pose/Shuttlecock engine
│   │   │   └── video_analysis_app.py       # Analysis orchestrator
│   │   └── templates/             # HTML Templates
│   │       ├── analyzer/          # App-specific templates
│   │       │   ├── analysis_detail.html
│   │       │   ├── base.html
│   │       │   ├── history.html
│   │       │   ├── index.html
│   │       │   └── upload.html
│   │       └── registration/      # Auth templates
│   │           ├── login.html
│   │           └── signup.html
│   ├── db.sqlite3                 # Development Database
│   ├── manage.py                  # Django management script
│   ├── media/                     # User-uploaded & generated media
│   │   ├── models/                # ML model files (e.g., shuttlecock.pt)
│   │   ├── outputs/               # Generated annotated videos
│   │   ├── videos/                # Original uploaded videos
│   │   └── frames/                # Extracted key frame images
│   └── yolo11n-pose.pt            # Pose model file (in media/models/)
├── implementation_phase.md        # External report document
├── migration_log.md               # External report document
├── pose_estimation_documentation.md # External report document
└── PROJECT_DOCUMENTATION.md       # This file
```

## 4. Core Component Implementation Details

### 4.1. Django Configuration (`badminton/badminton/`)

-   **`settings.py`**: Defines global project settings. Key configurations include:
    -   `INSTALLED_APPS`: Registers the `analyzer` app alongside default Django apps.
    -   `DATABASES`: Configured for SQLite initially.
    -   `MEDIA_ROOT` & `MEDIA_URL`: Define filesystem path and URL prefix for user uploads and generated files (videos, frames, models).
    -   `STATIC_ROOT` & `STATIC_URL`: For serving static assets like CSS/JS.
    -   `AUTH_PASSWORD_VALIDATORS`, `LOGIN_REDIRECT_URL`, `LOGOUT_REDIRECT_URL`: Standard Django authentication settings.
    -   `YOLO_MODELS_DIR`: Custom setting pointing to `media/models/` for loading ML models.
-   **`urls.py`**: Root URL configuration. Delegates most paths (`""`) to `analyzer.urls`, includes Django auth URLs (`/accounts/`), a custom signup URL (`/accounts/signup/`), and the admin interface (`/admin/`). Includes development-only static/media file serving.

### 4.2. Analyzer Application (`badminton/analyzer/`)

-   **`models.py`**:
    -   `VideoAnalysis`: Central model storing all data per analysis task.
        -   Fields: `user` (ForeignKey to `User`), `title`, `created_at`, `updated_at`, `input_video` (FileField), `output_video` (FileField, nullable), `hls_path` (CharField, nullable), `shot_type` (CharField with choices), `process_interval` (PositiveIntegerField), `source_fps` (FloatField, nullable), `is_analyzed` (BooleanField), `analysis_summary` (JSONField, nullable), `pre_hit_frame`, `hit_frame`, `follow_through_frame` (ImageFields, nullable).
        -   Includes `video_upload_path` and `output_upload_path` helpers for unique filenames.
        -   Overrides `delete()` method to ensure associated media files are deleted from storage when a record is removed.
-   **`views.py`**: Handles HTTP request logic.
    -   Standard views (`home`, `signup`, `upload_video`, `analysis_detail`, `analysis_history`, `download_analysis`, `delete_analysis`) manage user interaction, form processing, data retrieval, and template rendering.
    -   Uses `@login_required` decorator to protect relevant views.
    -   **Asynchronous Analysis Handling:**
        -   `start_analysis`: Triggered by AJAX POST. Creates `DjangoVideoAnalysisApp` instance, stores it in a global `running_analyses` dict, and launches `analyzer.analyze_video()` in a background `threading.Thread` via `run_analysis_thread`. Returns immediately with `{'status': 'started'}`.
        -   `run_analysis_thread`: Executes the analysis in the background, handles completion/errors, updates status in `running_analyses`.
        -   `analysis_progress`: AJAX GET endpoint polled by the frontend. Checks `running_analyses` or `analysis.is_analyzed` to return current progress and status as JSON.
        -   `stop_analysis`: AJAX POST endpoint. Calls `analyzer.request_stop()` on the running instance.
-   **`forms.py`**:
    -   `SignUpForm`: Inherits `UserCreationForm` for user registration.
    -   `VideoUploadForm`: `ModelForm` based on `VideoAnalysis` for handling video uploads and analysis parameter selection (shot type, title, interval, fps).
-   **`urls.py`**: Defines app-specific URL patterns with namespaces (`app_name = 'analyzer'`) and named routes (e.g., `name='analysis_detail'`). Includes routes for standard pages and AJAX endpoints.
-   **`templates/`**: Contains HTML templates using Django Template Language and Bootstrap CSS. Includes a `base.html` for inheritance and specific templates for each view.

### 4.3. Analysis Services (`badminton/analyzer/services/`)

This layer contains the core computational logic, decoupled from Django's web components.

-   **`video_analysis_app.py` (`DjangoVideoAnalysisApp` class):** Orchestrates the analysis workflow.
    -   **Initialization:** Takes the `VideoAnalysis` model instance.
    -   **State Management:** Tracks progress (`_progress`), status messages (`_status_message`), key frames (`_key_frames`), stop requests (`_stop_requested`).
    -   **Core Method (`analyze_video`)**:
        1.  Initializes `BadmintonVideoTracker`.
        2.  Opens input video (`cv2.VideoCapture`).
        3.  Reads all frames into memory (performance optimization V5).
        4.  Sets up `ffmpeg` subprocess for MP4 output encoding, piping frames via stdin.
        5.  Iterates through frames:
            - Checks `_stop_requested` flag.
            - Implements selective frame processing based on `process_interval` or if key frames are missing (performance optimization V5).
            - Calls `self.process_frame()` for analysis.
            - Writes annotated or original frame (encoded as PNG) to `ffmpeg` stdin pipe.
            - Implements early termination if all key frames are found (performance optimization V5).
        6.  Finalizes `ffmpeg` encoding.
        7.  Retrieves summary from tracker (`tracker.get_technique_summary()`), ensuring JSON serializability.
        8.  Saves key frame images (`save_frame_as_image` helper) and output video path to the `VideoAnalysis` model.
        9.  Saves JSON summary and updates `is_analyzed` flag on the model.
        10. Returns completion status.
    -   **Helper Methods:** `initialize_tracker`, `process_frame` (calls tracker), `save_frame_as_image`, `request_stop`.

-   **`badminton_pose_estimator.py` (`BadmintonVideoTracker` class):** The core analysis engine.
    -   **Initialization:**
        -   Loads YOLOv11-Pose and custom YOLO shuttlecock models using `ultralytics.YOLO`.
        -   Moves models to GPU (`cuda`) if available (performance optimization V5).
        -   Initializes state: `target_shot_type`, `tracked_id`, `key_frames` dict, `phase_detections` dict, `min_shuttle_distance`, `sequence_state`.
        -   Caches `SHOT_THRESHOLDS` for the target shot.
    -   **Detection Methods:**
        -   `detect_shuttlecock`: Uses shuttlecock model. Implements ROI tracking and top-half frame optimization (performance optimization V5).
        -   Pose detection is done within `process_video_frame` using `self.pose_model`.
    -   **Biomechanical Calculations:**
        -   `calculate_grip_angle`: Calculates elbow angle using right shoulder, elbow, wrist keypoints (vector math).
        -   `calculate_shoulder_angle`: Calculates upper arm angle relative to vertical using right shoulder, elbow keypoints (`arctan2`).
        -   `calculate_shuttle_distance`: Euclidean distance between wrist and shuttlecock center.
    -   **Core Frame Processing (`process_video_frame` -> `analyze_frame`):**
        1.  Runs pose detection.
        2.  Updates tracked player (`update_tracked_player`).
        3.  Detects shuttlecock.
        4.  Calculates angles and wrist-shuttle distance.
        5.  Tracks minimum wrist-shuttle distance and the corresponding frame (`min_distance_frame`) for accurate hit detection (implementation V_UserMgmt_Hit).
        6.  Determines shot phase (`determine_shot_phase_improved`) using a state machine considering angles, thresholds, and sequence (pre-hit -> hit -> follow-through) (implementation V_UserMgmt_Hit).
        7.  Stores frame data in `phase_detections` list based on determined phase.
        8.  Updates `key_frames` dictionary using `update_key_frames_from_detections` logic (selects representative frame for each phase, using `min_distance_frame` for 'hit') (implementation V4 & V_UserMgmt_Hit).
        9.  Generates annotated frame using `visualize_analysis`.
        10. Returns annotated frame and analysis data.
    -   **Visualization (`visualize_analysis`):** Uses OpenCV (`draw_skeleton`, `draw_shuttlecock`, `draw_info_overlay`) to draw skeleton (colored by phase), shuttlecock box, info text (phase, angles, threshold match), and wrist-shuttle distance line/text onto the frame.
    -   **Summary Generation (`get_technique_summary`):** Compiles final results, calculates timing between key frames, ensures data (keypoints, boxes) is converted to lists for JSON serialization. Returns status ('complete', 'incomplete', 'error').

## 5. Implementation Evolution: From Gradio Prototype to Django Plan

The project's implementation journey involved significant evolution, starting with a Gradio prototype and culminating in a plan for a full-featured Django application.

-   **Initial Gradio Prototype (V1-V3):**
    -   **Purpose:** Rapidly validate the core computer vision concepts (pose estimation, angle calculation, basic phase/technique identification) and get early user feedback.
    -   **Implementation:** Python functions performing analysis were directly mapped to Gradio interface components (`gr.Video`, `gr.Image`, `gr.Textbox`, `gr.Radio`). Gradio handled the UI generation and basic input/output flow.
    -   **Key Developments:** Initial pose estimation (V1), video processing (V2), introduction of phase detection concepts, key frame display, and saving annotated output video (V3). Shuttlecock detection was added conceptually and later implemented with a trained model.
    -   **Limitations:** Lack of persistent storage (analysis results lost after session), no user accounts or history, limited UI customization, potential blocking behavior during long analysis tasks, and less scalable architecture.

-   **Refinements within Gradio (V4-V5):**
    -   **Focus:** Improving the accuracy and performance of the core analysis engine based on testing.
    -   **Key Developments:** Refined key frame selection logic (V4), implemented crucial performance optimizations (GPU acceleration, selective frame processing, ROI/top-half shuttle detection, early termination, loading all frames first), added UI controls for performance settings and a stop button (V5).

-   **Hit Detection & State Machine (V_UserMgmt_Hit):**
    -   **Focus:** Enhancing the analysis core with more sophisticated hit detection and phase sequencing.
    -   **Key Developments:** Implemented hit frame identification based on minimum wrist-shuttlecock distance. Developed a state machine (`determine_shot_phase_improved`) for robust phase sequence detection. Added distance visualization.

-   **Planned Migration to Django:**
    -   **Motivation:** To overcome Gradio's limitations and build a complete, persistent, and scalable web application. Key drivers included the need for:
        -   **User Management:** Secure user registration, login, and association of analyses with specific users.
        -   **Data Persistence:** Storing uploaded videos, analysis parameters, generated results (output videos, key frames, summaries), and user history reliably in a database (SQLite initially).
        -   **Scalable Architecture:** Django's MVT pattern provides better separation of concerns (data, logic, presentation) compared to Gradio's direct function-to-UI mapping.
        -   **Customizable UI/UX:** Full control over the frontend using HTML, CSS, and JavaScript for a tailored user experience beyond Gradio's standard components.
        -   **Robust Asynchronous Task Handling:** Implementing background analysis using threading (as shown in `views.py`) or potentially more robust solutions like Celery within the Django framework to prevent blocking web requests.
    -   **Transition Strategy:**
        1.  **Structure:** Set up the standard Django project and app structure (`manage.py`, `settings.py`, `models.py`, `views.py`, `urls.py`, `templates/`, `static/`).
        2.  **Core Logic Reuse:** Encapsulate the refined Python analysis logic (from `badminton_pose_estimator.py` and the core processing parts initially in the Gradio app's main script) into dedicated service classes (`DjangoVideoAnalysisApp`, `BadmintonVideoTracker`) within the Django app's `services/` directory.
        3.  **Data Modeling:** Define Django models (`VideoAnalysis` in `models.py`) to represent the data previously handled transiently or saved manually in Gradio.
        4.  **Views & URLs:** Create Django views (`views.py`) to handle HTTP requests, interact with models, call the analysis services, and render templates. Define URL patterns (`urls.py`) to map web addresses to these views.
        5.  **Forms:** Use Django Forms (`forms.py`) for handling user input (video upload, signup) with validation.
        6.  **Templates:** Build the HTML frontend (`templates/`) using Django's template language, replacing Gradio's auto-generated UI. Implement AJAX for asynchronous communication (start analysis, progress polling).
        7.  **Authentication:** Leverage Django's built-in authentication system.

-   **Technology Selection Revisited:** The choice of YOLOv11-Pose and the custom shuttlecock model remained consistent. Django was confirmed as the target framework for its comprehensive features. FFmpeg continued to be used for reliable video encoding within the Django service layer.

## 6. Machine Learning Model Details (Consistent Across Phases)

-   **Pose Model (`yolo11n-pose.pt`):** A YOLOv11-nano model fine-tuned for pose estimation. It outputs keypoint coordinates (x, y) and confidence scores for detected human joints. Loaded and run via the Ultralytics library.
-   **Shuttlecock Model (`shuttlecock.pt`):** A YOLO model trained specifically (using Roboflow) to detect badminton shuttlecocks. Training involved a dataset of 325 images. Achieved good mAP50 score, indicating effectiveness in detecting shuttlecocks, though high-precision localization (mAP50-95) was more challenging. Loaded and run via the Ultralytics library.

---
*This document describes the technical implementation of the Badminton Technique Analyzer project, detailing its evolution from a Gradio prototype to the final planned Django architecture.*
