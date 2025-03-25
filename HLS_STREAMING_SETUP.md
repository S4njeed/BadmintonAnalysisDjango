# HLS Streaming Implementation Guide

This guide explains how we've replaced AV1 codec with HLS (HTTP Live Streaming) for more efficient video delivery in the Badminton Analyzer application.

## Changes Made

1. **Model Update**:
   - Added `hls_path` field to the `VideoAnalysis` model
   - Updated the model's `delete` method to clean up HLS segments when an analysis is deleted

2. **Video Processing**:
   - Replaced AV1 encoding with HLS encoding using FFmpeg
   - Added HLS playlist and segment generation
   - Created separate storage for HLS streams under `media/hls/`

3. **Frontend Updates**:
   - Added HLS.js for client-side HLS playback
   - Updated Plyr.js configuration to handle HLS streams
   - Added fallback to MP4 if HLS streaming isn't supported

4. **Nginx Configuration**:
   - Created a configuration file for serving HLS content with proper headers

## Setup Instructions

### 1. Apply Django Migrations

First, you need to apply migrations to update the database schema:

```bash
python manage.py makemigrations analyzer
python manage.py migrate
```

### 2. Configure Nginx

1. Install Nginx if not already installed:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install nginx
   
   # CentOS/RHEL
   sudo yum install nginx
   
   # Windows
   # Download and install from nginx.org
   ```

2. Copy the provided Nginx configuration:
   ```bash
   # Edit paths in nginx_hls_config.conf to match your setup
   # Replace /path/to/your/project/ with your actual project path
   
   # For Ubuntu/Debian/Linux
   sudo cp nginx_hls_config.conf /etc/nginx/sites-available/badminton
   sudo ln -s /etc/nginx/sites-available/badminton /etc/nginx/sites-enabled/
   
   # For Windows
   # Copy to the nginx/conf directory
   ```

3. Test the configuration:
   ```bash
   # Linux
   sudo nginx -t
   
   # Windows
   nginx -t
   ```

4. Restart Nginx:
   ```bash
   # Linux
   sudo systemctl restart nginx
   
   # Windows
   nginx -s reload
   ```

### 3. Ensure FFmpeg Is Installed

FFmpeg is required for HLS encoding:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### 4. Create Required Directories

Ensure the HLS directory exists:

```bash
mkdir -p badminton/media/hls
```

## How HLS Streaming Works

1. When a video is analyzed, FFmpeg generates:
   - A master playlist (.m3u8) file
   - Multiple video segments (.ts) files

2. Nginx serves these files with proper MIME types and caching directives

3. In the browser, HLS.js loads the playlist and fetches segments as needed

4. Benefits include:
   - Adaptive bitrate streaming
   - Better compatibility across devices
   - Reduced initial loading time
   - Lower bandwidth usage

## Troubleshooting

- If videos don't play, check browser console for HLS.js errors
- Verify that Nginx is serving .m3u8 and .ts files with correct MIME types
- Ensure the Django media URL is correctly configured
- Check that FFmpeg is installed and available in PATH

## Additional Notes

- For production deployment, use the commented HTTPS configuration
- Consider implementing authentication for video access if needed
- The current implementation provides a simple VOD (Video on Demand) HLS setup