# Migration Log: Gradio to Django

This document outlines the key changes and architectural shifts involved in migrating the badminton analysis application from a Gradio-based implementation to a Django-based web application.

## 1. Framework Shift

*   **Old (Gradio):** The application was initially built using Gradio, a Python library primarily designed for creating quick machine learning demos and interfaces. It simplified UI creation for ML models but offered less flexibility for building full-featured web applications.
*   **New (Django):** The application now uses Django, a high-level Python web framework that encourages rapid development and clean, pragmatic design. Django provides a robust structure (Models-Views-Templates), an ORM, authentication, admin interface, and more, enabling a more scalable and maintainable application.

## 2. User Interface (UI)

*   **Old (Gradio):** The UI was likely defined programmatically using Gradio's components (`gr.Interface`, `gr.Video`, `gr.Textbox`, etc.). UI logic was tightly coupled with the backend processing function.
*   **New (Django):** The UI is now separated using Django's templating engine.
    *   **Templates (`analyzer/templates/`):** HTML files define the structure and presentation of pages (e.g., `index.html`, `upload.html`, `analysis_detail.html`, `base.html`). Jinja-like template tags (`{% %}`, `{{ }}`) are used to embed dynamic content and logic.
    *   **Forms (`analyzer/forms.py`):** Django Forms handle user input validation and rendering of HTML forms (e.g., `VideoUploadForm`, potentially user registration/login forms).

## 3. Backend Logic

*   **Old (Gradio):** The core video processing and analysis logic was likely contained within a single Python function passed to the `gr.Interface`. State management could be challenging for complex interactions.
*   **New (Django):** Backend logic is organized following Django's MVT pattern:
    *   **Views (`analyzer/views.py`):** Handle incoming HTTP requests, interact with models and services, and render templates. Functions like `upload_video`, `display_analysis`, `user_signup`, `user_login` likely reside here.
    *   **Services (`analyzer/services/`):** Encapsulate the core ML/analysis logic (e.g., `badminton_pose_estimator.py`, `video_analysis_app.py`), keeping it separate from the web request handling in views. This promotes modularity and testability.
    *   **URLs (`analyzer/urls.py`, `badminton/urls.py`):** Map URL patterns to specific view functions, defining the application's endpoints.

## 4. Data Handling & Persistence

*   **Old (Gradio):** Data handling might have been basic, potentially relying on temporary file storage or simple in-memory structures, depending on the specific Gradio implementation. Persistent storage would require custom handling.
*   **New (Django):** Django's ORM (Object-Relational Mapper) manages database interactions.
    *   **Models (`analyzer/models.py`):** Define the database schema using Python classes (e.g., `VideoAnalysis` model storing video paths, analysis results, user associations, timestamps).
    *   **Database (`db.sqlite3`):** Django uses a database (SQLite by default, as seen here) to persistently store application data defined by the models.
    *   **Migrations (`analyzer/migrations/`):** Manage changes to the database schema over time.

## 5. Authentication & Authorization

*   **Old (Gradio):** Gradio has basic authentication features, but implementing robust user accounts and permissions often requires external libraries or custom solutions.
*   **New (Django):** Leverages Django's built-in authentication system.
    *   **User Model:** Django provides a standard `User` model.
    *   **Views/Forms:** Handles login (`registration/login.html`), signup (`registration/signup.html`), and logout logic.
    *   **Permissions:** Allows associating data (like `VideoAnalysis` results) with specific users (`ForeignKey` to User model) and controlling access based on login status.

## 6. Serving & Deployment

*   **Old (Gradio):** Typically served as a standalone application using Gradio's built-in server, often suitable for demos or internal tools.
*   **New (Django):** Served using a standard web server gateway interface (WSGI) or asynchronous server gateway interface (ASGI) (e.g., Gunicorn, uvicorn) behind a web server (like Nginx). This is standard practice for production web applications. Files like `wsgi.py` and `asgi.py` facilitate this.

## Summary

The migration from Gradio to Django represents a shift from a simple ML demo interface to a full-stack web application architecture. This provides greater scalability, maintainability, feature richness (like user accounts and persistent storage), and adherence to standard web development practices. The separation of concerns (UI, logic, data) is much clearer in the Django structure.
