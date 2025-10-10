import cv2
import os
import torch
from sympy.strategies.core import switch
from ultralytics import YOLO
from util.detection import detect_video, detect_image

model_path = "./models"


models = [m for m in os.listdir(model_path) if m.endswith(".pt")]


print("================= Dental Disease detection ===================")
print("Choose option:")
print("1. Video")
print("2. Image")
print("3. Camera")
print("4. Exit")
choice = input(">>>: ")

if choice == "1":
    print("Enter Video path")
    input_video_path = input(">>>: ")
    detect_video(input_video_path)
elif choice == "2":
    print("Enter Image path")
    input_video_path = input(">>>: ")
    detect_image(input_video_path)
elif choice == "3":
    print("Opening Camera")
    detect_video(0)
elif choice == "4":
    exit()

print("================= Dental Disease detection ===================")
print("Choose models:")
i = 1
for m in models:
    print(f"{i}. {m}")
    i += 1
model_choice = int(input(">>>: "))
# Load the YOLO model
model = YOLO(f'./models/{models[model_choice-1]}')


# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
# or another version of YOLOv8 (e.g., yolov8s.pt for small)
model = YOLO('./models/best.pt')

# Load the video file
input_video_path = 'video.mp4'
output_video_path = 'out.mp4'

# Open the video using OpenCV
video_capture = cv2.VideoCapture(0)

# Iterate over each frame
frame_count = 0
ret = True
while ret:
    ret, frame = video_capture.read()  # Read a frame
    results2 = model.track(frame, persist=True,  verbose=False)
    frame_ = results2[0].plot()
    cv2.imshow("Video Player", frame_)  # Display the frame

    if cv2.waitKey(25) & 0xFF == ord('q'):

        break

# Release resources
video_capture.release()
# out_video.release()
cv2.destroyAllWindows()
