# Badminton Technique Analyzer

This Django application analyzes badminton technique from uploaded videos using pose estimation and provides feedback.

## Requirements

All required Python packages are listed in the `requirements.txt` file.

To install the dependencies, navigate to the project's root directory (where `requirements.txt` is located) in your terminal and run:

```bash
pip install -r requirements.txt
```

You will also need to have the necessary YOLO models (`yolo11n-pose.pt` and `shuttlecock.pt`) available. By default, the application looks for these in the project root directory.

Additionally, ensure you have an Ollama instance running and accessible if you want to use the AI feedback feature. The model specified in `ollama_feedback_service.py` (currently `gemma3:4b`) should be available in your Ollama setup.

To use the AI feedback feature, you need to install Ollama and download the `gemma3:4b` model. Follow the instructions on the [Ollama website](https://ollama.com/download) to install Ollama. Once installed, open your terminal and run the following command to download the model:

```bash
ollama pull gemma3:4b
```

## Running the Server Locally

1.  **Navigate to the project directory:**
    Open your terminal and change the directory to where the `manage.py` file is located (the `badminton` directory within the project root).

    ```bash
    cd badminton
    ```

2.  **Apply database migrations (if necessary):**
    If this is the first time running the project or if there have been model changes, apply the database migrations:

    ```bash
    python manage.py migrate
    ```

    If you encounter errors in the console related to the database, you might need to create migrations first:

    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

3.  **Run the development server:**
    Start the Django development server using the following command:

    ```bash
    python manage.py runserver
    ```

4.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:8000/`.

## Hosting with Ngrok

Ngrok allows you to expose your local development server to the internet. This is useful for testing on different devices or sharing access temporarily.

1.  **Install Ngrok:**
    Download and install ngrok from the official website: [https://ngrok.com/](https://ngrok.com/)

2.  **Run the Django Server:**
    Make sure your Django development server is running locally (as described in the "Running the Server Locally" section). By default, it runs on port 8000.

3.  **Expose the local server with Ngrok:**
    Open a *new* terminal window and run the following command, replacing `8000` if your server is running on a different port:

    ```bash
    ngrok http 8000
    ```

4.  **Get the Ngrok URL:**
    Ngrok will display a public URL (e.g., `https://<random-string>.ngrok-free.app`). This URL now points to your local Django server.

5.  **Update Django Settings (Important):**
    For Ngrok to work correctly with Django, you need to add the Ngrok URL to your `ALLOWED_HOSTS` in `badminton/badminton/settings.py`.

    Open `badminton/badminton/settings.py` and add your Ngrok hostname (e.g., `<random-string>.ngrok-free.app`) to the `ALLOWED_HOSTS` list:

    ```python
    ALLOWED_HOSTS = ["127.0.0.1", "<random-string>.ngrok-free.app"]
    ```

    You might also need to add it to `CSRF_TRUSTED_ORIGINS` if you encounter CSRF issues, especially with forms:

    ```python
    CSRF_TRUSTED_ORIGINS = ["https://<random-string>.ngrok-free.app"]
    ```

    **Remember to restart your Django server** after modifying `settings.py` for the changes to take effect, this should happen automatically.

6.  **Access via Ngrok URL:**
    You can now access your application using the public Ngrok URL provided in the terminal.
