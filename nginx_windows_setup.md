# Setting up Nginx for HLS Streaming in Windows

This guide will help you set up Nginx to serve HLS streams from your Badminton Analyzer application.

## 1. Install the Updated Configuration

1. Copy the updated configuration file to your Nginx conf directory:

```cmd
copy nginx_windows_updated.conf C:\tools\nginx-1.27.4\conf\nginx.conf
```

> **Note**: This will replace your main nginx.conf file. If you want to keep your existing setup, you can create a conf.d directory and include it instead:
> ```cmd
> mkdir C:\tools\nginx-1.27.4\conf\conf.d
> copy nginx_windows_updated.conf C:\tools\nginx-1.27.4\conf\conf.d\badminton.conf
> ```
> Then add this line inside the `http` block in your main nginx.conf:
> ```
> include conf.d/*.conf;
> ```

## 2. Test the Configuration

```cmd
cd C:\tools\nginx-1.27.4
nginx -t
```

You should see:
```
nginx: the configuration file C:\tools\nginx-1.27.4\conf\nginx.conf syntax is ok
nginx: configuration file C:\tools\nginx-1.27.4\conf\nginx.conf test is successful
```

## 3. Start or Reload Nginx

If Nginx is not running:

```cmd
cd C:\tools\nginx-1.27.4
start nginx
```

If Nginx is already running:

```cmd
cd C:\tools\nginx-1.27.4
nginx -s reload
```

## 4. Test the HLS Stream

1. Make sure your Django development server is running:
   ```cmd
   cd C:\Users\sanje\OneDrive\Documents\Computer Science\cLiendD\badminton
   python manage.py runserver
   ```

2. Generate a new video analysis through your application interface

3. Visit the analysis detail page in your browser and check if the video plays using HLS streaming

4. You can also directly access an HLS stream URL:
   ```
   http://localhost/media/hls/analysis_XX/playlist.m3u8
   ```
   (Replace XX with an actual analysis ID)

## Troubleshooting

### Check if Nginx is Running
```cmd
tasklist | findstr nginx
```

### View Nginx Error Logs
```cmd
type C:\tools\nginx-1.27.4\logs\error.log
```

### Common Issues

1. **403 Forbidden Error**: Make sure the path in the nginx.conf file matches your actual project directory

2. **404 Not Found Error**: Verify that the HLS directory exists and contains .m3u8 and .ts files

3. **MIME Type Error**: If the browser can't play the HLS stream, check that the MIME types are correctly set in the Nginx configuration

4. **Permission Issues**: Make sure Nginx has read permissions on your media directory

5. **Port Conflicts**: Check if port 80 is already in use by another application

### Testing with curl

You can use curl to check if Nginx is serving the files correctly:

```cmd
curl -I http://localhost/media/hls/analysis_XX/playlist.m3u8
```

You should see content-type: application/vnd.apple.mpegurl in the response headers.

## Stopping Nginx

When you're done testing:

```cmd
cd C:\tools\nginx-1.27.4
nginx -s stop