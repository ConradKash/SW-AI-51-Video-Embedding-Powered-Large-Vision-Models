import cv2
import os
import torch
from sympy.strategies.core import switch
from ultralytics import  YOLO
from  util.detection import detect_video, detect_image

model_path ="./models"


models = [ m for m in os.listdir(model_path) if m.endswith(".pt")]


print ("================= Dental Disease detection ===================")
print ("Choose option:")
print ("1. Video")
print ("2. Image")
print ("3. Camera")
print ("4. Exit")
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

print ("================= Dental Disease detection ===================")
print ("Choose models:")
i=1
for m in models:
    print (f"{i}. {m}")
    i+=1
model_choice = int(input(">>>: "))
# Load the YOLO model 
model = YOLO(f'./models/{models[model_choice-1]}')



