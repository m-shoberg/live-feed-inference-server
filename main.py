import sys
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

# Modified VideoWindow class for Flask streaming
class VideoWindow:

    def __init__(self, video_source):
        self.video_source = video_source
        self.current_frame = None

    def update_frame(self, frame):
        self.current_frame = frame

    def get_frame(self):
        if self.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', self.current_frame)
            return buffer.tobytes()
        else:
            return None

# Function to generate frames for Flask streaming
def gen_frames(video_window):
    while True:
        frame = video_window.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

# Flask routes for streaming
@app.route('/camera_feed/<int:camera_id>')
def camera_feed(camera_id):
    return Response(gen_frames(video_windows[camera_id]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to run tracker in a thread
def run_tracker_in_thread(filename, model, file_index, video_window):
    video = cv2.VideoCapture(filename)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()

        # Update the frame in the video window
        video_window.update_frame(res_plotted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()

# Function to start video processing threads
def start_video_processing():
    # Create VideoWindow instances
    global video_windows
    video_windows = [VideoWindow("http://195.196.36.242/mjpg/video.mjpg"), 
                     VideoWindow("http://173.198.10.174//mjpg/video.mjpg"), 
                     VideoWindow("http://217.171.212.63/mjpg/video.mjpg")]

    # Load the models
    model1 = YOLO('yolov8n.pt')
    model2 = YOLO('yolov8n-seg.pt')
    model3 = YOLO('yolov8n-seg.pt')

    # Create and start tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=("http://195.196.36.242/mjpg/video.mjpg", model1, 0, video_windows[0]), daemon=True)
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=("http://173.198.10.174//mjpg/video.mjpg", model2, 1, video_windows[1]), daemon=True)
    tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=("http://217.171.212.63/mjpg/video.mjpg", model3, 2, video_windows[2]), daemon=True)
    
    tracker_thread1.start()
    tracker_thread2.start()
    tracker_thread3.start()

# Main entry point
if __name__ == '__main__':
    # Start video processing in separate threads
    start_video_processing()

    # Start Flask app in the main thread
    app.run(debug=True, threaded=True, port=5000)
