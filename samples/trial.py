import cv2
import torch
from ultralytics import  YOLO

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./models/best.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)

# Load the video file
input_video_path = 'video.mp4'
output_video_path = 'out.mp4'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(input_video_path)

# Iterate over each frame
frame_count = 0
ret=True
while ret: 
    ret, frame = video_capture.read()  # Read a frame   
    results2 = model.track(frame,persist=True,  verbose=False)
    frame_ = results2[0].plot()
    cv2.imshow("Video Player", frame_)  # Display the frame

    if cv2.waitKey(25) & 0xFF == ord('q'):

        break
    
# Release resources
video_capture.release()
#out_video.release()
cv2.destroyAllWindows()

