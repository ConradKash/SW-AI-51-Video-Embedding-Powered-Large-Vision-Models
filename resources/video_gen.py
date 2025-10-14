#THIS PROGRAM IS USED TO GENERATE A VIDEO FROM IMAGES
import cv2 as cv
import numpy as np
import os

images_path = r"C:\Users\Administrator\Downloads\Test images"
video_out_path=".\\outvideo.mp4"
image_ext=[".jpg", ".jpeg", ".png"]

images = []
height, width = 0, 0   #defines the max width and height of the images (WxL)


for f in os.listdir(images_path):
    if os.path.splitext(f)[1].lower() in image_ext:
        images.append(f)
        im = cv.imread(os.path.join(images_path, f))
        if height < im.shape[0]:
            height, width, _ = im.shape
        # if shape[1] < im.shape[1]:
        #     shape[1] = im.shape[1]   



#VIDEO PROPOERTIES
FPS = 2
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec
out_video = cv.VideoWriter(video_out_path,fourcc,FPS,(width,height))
for f in images:
    im = cv.imread(os.path.join(images_path, f))
   # Resize to match the first image’s size
    if (im.shape[1], im.shape[0]) != (width, height):
        im = cv.resize(im, (width, height))
    out_video.write(im)

out_video.release()
print("Colpleted")
