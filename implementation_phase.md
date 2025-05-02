# Implementation Phase Logbook - Badminton Analysis Project

This logbook documents the changes made during the implementation phase, focusing on user-specific history and deletion features.

## Feature: User-Specific Analysis History and Deletion

**Date:** 2025-04-19

**Objective:** Modify the application so that users can only see their own analysis history and have the ability to delete their own analyses.

**Changes:**

1.  **Model Update (`badminton/analyzer/models.py`):**
    *   Added a `ForeignKey` field named `user` to the `VideoAnalysis` model, linking it to the standard Django `User` model (Lines 26-33).
    *   Imported `django.contrib.auth.models.User` (Line 6).
    *   Set `on_delete=models.CASCADE` to ensure analyses are deleted if the user is deleted.
    *   Allowed `null=True` and `blank=True` for potential future handling of anonymous uploads or existing data.

2.  **Database Migrations:**
    *   Generated a new migration file using `python manage.py makemigrations analyzer`.
    *   Applied the migration to update the database schema using `python manage.py migrate`.

3.  **View Updates (`badminton/analyzer/views.py`):**
    *   **`upload_video` (Lines 40-53):**
        *   Added `@login_required` decorator (Line 39) to ensure only logged-in users can upload.
        *   Modified the form saving logic to use `commit=False`, set `analysis.user = request.user`, and then call `analysis.save()` (Lines 46-48) to associate the uploaded analysis with the current user.
        *   Imported `django.contrib.auth.decorators.login_required` (Line 10).
    *   **`analysis_history` (Lines 66-70):**
        *   Added `@login_required` decorator (Line 65) to restrict access to logged-in users.
        *   Modified the queryset to filter by the current user: `VideoAnalysis.objects.filter(user=request.user)` (Line 69).
    *   **`delete_analysis` (Lines 211-225):**
        *   Created a new view function `delete_analysis`.
        *   Added `@login_required` decorator (Line 210).
        *   Ensured the request method is POST (Lines 213-215).
        *   Fetched the `VideoAnalysis` object using `get_object_or_404`, crucially including `user=request.user` in the query to prevent users from deleting others' analyses (Line 218).
        *   Called `analysis.delete()` within a try-except block (Lines 220-224).
        *   Added success/error messages using `django.contrib.messages`.
        *   Redirected back to the history page upon completion or error.

4.  **URL Configuration (`badminton/analyzer/urls.py`):**
    *   Added a new URL pattern for the `delete_analysis` view: `path('delete/<int:analysis_id>/', views.delete_analysis, name='delete_analysis')` (Line 24).

5.  **Template Update (`badminton/analyzer/templates/analyzer/history.html`):**
    *   Added a small form within the table row for each analysis (Lines 48-54).
    *   The form submits via POST to the `delete_analysis` URL.
    *   Included a `{% csrf_token %}` for security.
    *   Added a submit button styled as a delete button (`btn-outline-danger`).
    *   Included an `onclick` JavaScript confirmation dialog (`return confirm(...)`) to prevent accidental deletions.

**Outcome:** The analysis history page now displays only the analyses belonging to the currently logged-in user. Users can delete their own analyses via a button on the history page, but cannot delete analyses belonging to other users. Uploads are now correctly associated with the user performing the upload.


## Feature: Basic User Authentication Flow

**Date:** 2025-04-19

**Objective:** Add standard Django login/logout functionality to allow users to authenticate.

**Changes:**

1.  **Include Auth URLs (`badminton/badminton/urls.py`):**
    *   Added `path("accounts/", include("django.contrib.auth.urls"))` to the main `urlpatterns` (Line 23) to incorporate Django's built-in authentication views (login, logout, password reset, etc.).

2.  **Configure Settings (`badminton/badminton/settings.py`):**
    *   Added `LOGIN_REDIRECT_URL = '/'` (Line 130) to redirect users to the home page after successful login.
    *   Added `LOGOUT_REDIRECT_URL = '/'` (Line 131) to redirect users to the home page after logging out.

3.  **Create Login Template (`badminton/analyzer/templates/registration/login.html`):**
    *   Created a new template file at the path expected by `django.contrib.auth.urls`.
    *   The template extends `analyzer/base.html`.
    *   Includes a standard login form with fields for username and password, CSRF token, and handling for form errors and the `next` parameter.
    *   Added basic styling for form inputs within a `{% block extra_css %}`.

4.  **Update Base Template (`badminton/analyzer/templates/analyzer/base.html`):**
    *   Added conditional logic within the navigation bar (`ul.navbar-nav`) (Lines 191-216).
    *   If `user.is_authenticated`, display a dropdown with the username and a "Logout" link (which submits a hidden form).
    *   If the user is not authenticated, display a "Login" link pointing to `{% url 'login' %}`.

**Outcome:** Users can now navigate to `/accounts/login/` to log in. The navigation bar dynamically shows either a "Login" link or the username with a "Logout" option based on authentication status. Login-protected views (like history and upload) will automatically redirect unauthenticated users to the login page.
