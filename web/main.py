import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO

model_path = os.path.join(os.path.dirname(__file__), "../models/")


st.title("Dental Diseases Detection App")
#st.header("Welcome to Dental Diseases Detection Web App")

models = [ m for m in os.listdir(model_path) if m.endswith(".pt")]

model_opn = st.selectbox(
    "Choose model",
    models
)

#SELECT MODEL FROM OPTION
if model_opn is not None:
    model = YOLO(os.path.join(model_path,model_opn))



option = st.selectbox(
    "Choose Interaction Mode",
    ("Image", "Video", "Web Cam"),
)

file = None
cam_enabled = False 
#upload file
if option == "Image" or option == "Video":
    file = st.file_uploader(f"Upload {option}", type=["jpg", "png", "jpeg", "mp4"])

    
    if file is not None:
        image = Image.open(file)

        #CONVERT IMAGE TO NP ARRAY FOR PROCESSING IN CV2 AND YOLO
        np_image = np.array(image)
        cv_image = cv2.cvtColor(np_image,cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded image", width="stretch")
            #st.button("Predict", type="primary", on_click=None)
        with col2:
            #PREDICTION COLUM THAT CONTAINS IMAGES THAT HAVE A PROBLEM
            pred_result= model.track(cv_image)
            pred_image = pred_result[0].plot()
            #nOW TAKE IT BACK TO PILOW FOR DISPLAY
            pil_pred = Image.fromarray(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))

            st.image(pil_pred, caption="Detected image", width="stretch")
            st.button("Save", type="primary", on_click=None)

if option == "Web Cam":
    cam_enabled = True
    cam = st.camera_input("Video Feed", disabled=not cam_enabled)

    if cam is not None:
        cam_bytes = cam.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(cam_bytes, np.uint8), cv2.IMREAD_COLOR)

        result=model.track(cv_image)
        pred_image = result[0].plot()
        
        #Convert to pilow

        #nOW TAKE IT BACK TO PILOW FOR DISPLAY
        pil_pred = Image.fromarray(cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB))
        st.image(pil_pred, caption="Detected Image")


 



def predict(img):


    return img

