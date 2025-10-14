# SW-AI-51: Video Embedding Powered Large Vision Models for Dental Disease Detection in Uganda

## Overview
This project, **SW-AI-51**, explores the use of **AI-powered vision models** for **automatic detection of dental diseases** in patients through **video embeddings and large vision models (LVMs)**.  
It is designed to assist dental professionals in Uganda by providing an **AI-assisted diagnostic tool** capable of analyzing intraoral videos and images to identify potential dental conditions.

The system leverages **deep learning**, **computer vision**, and **embedding techniques** to extract meaningful features from dental imagery and predict specific oral health issues efficiently.

---

## Keywords
- Large Vision Models (LVMs),
- Dentistry
- Video Embeddings
- Dental Disease
- Detection and Localization
- Explainable AI
- Resource-Constrained Screening

---

## Objectives
- To develop a **large vision model** capable of identifying various **dental diseases** from video input.
- To use **video embedding representations** for accurate feature extraction and disease classification.
- To provide a **scalable and accessible solution** for dental diagnostics in **resource-limited healthcare settings**.
- To support **local dental practitioners in Uganda** with AI-driven insights for early detection and improved treatment planning.

---

## Features
- **AI-based disease detection** from dental videos and images.  
- **Video embedding pipeline** for frame-level feature extraction.  
- **Transfer learning** using state-of-the-art large vision models (e.g., ViT, CLIP, or YOLO variants).  
- **Prediction and visualization dashboard** for dental conditions.  
- **Localized for Uganda** – trained on regional datasets for improved accuracy on local cases.

---

## System Architecture
1. **Data Acquisition** – Intraoral videos and images are collected from dental clinics.  
2. **Preprocessing** – Frames are extracted, cleaned, and normalized for analysis.  
3. **Embedding Generation** – Each frame is passed through a pretrained large vision model to obtain embeddings.  
4. **Classification Layer** – Fine-tuned neural network predicts the dental condition.  
5. **Visualization & Reporting** – Results displayed with disease probability and visual highlights.

---

## Getting Started

### Run the Project in Google Colab
You can explore and run the model directly on Google Colab:  
 [Open in Colab](https://colab.research.google.com/drive/12cvlQp-xRTsQ4L5cA1WOv27YzZcg10Qu#scrollTo=O-ohMJKPVDDM)

### Requirements
If running locally, ensure you have the following installed:

```bash
python>=3.8
torch>=2.0.0
torchvision
ultralytics
numpy
opencv-python
matplotlib
scikit-learn
pandas
```
or install in the commandline
```bash
pip install -r requirements.txt
```

## Definitions of Dental Classes
1. Hypodonthia

Hypodontia is the developmental absence of one or more teeth, excluding third molars, and is one of the most common dental anomalies. It can be caused by a combination of genetic and environmental factors and is often associated with other syndromes like ectodermal dysplasia or Down's syndrome

![Hypodonthia](resources\hypodonthia.jpg)

2. Tooth Discoloration
Teeth discoloration is caused by stains from food, drinks, and tobacco (extrinsic) and internal factors like aging, trauma, medications, and certain illnesses

![Tooth Discoloration](resources\ToothDiscoloration.jpg)

3. Mouth Ulcer
A mouth ulcer is a painful sore, or ulcer, on the lining of the mouth that can appear on the cheeks, lips, gums, or tongue

![Tooth Discoloration](resources\ulcer.jpg)

4. Gingivitis
Gingivitis is a reversible, early form of gum disease causing inflamed, red, swollen, and bleeding gums, often due to poor oral hygiene leading to plaque buildup

![Tooth Discoloration](resources\Gingivitis.jpg)

5. Caries
Dental caries, or tooth decay, is the destruction of tooth hard tissue by acids produced by plaque bacteria feeding on sugars. It starts as a white spot, progressing to brown or black spots and then cavities as acid erodes enamel.

![Caries](resources\caries.jpg)

6. Calculus
In dentistry, calculus or tartar is a form of hardened dental plaque. It is caused by precipitation of minerals from saliva and gingival crevicular fluid in plaque on the teeth.

![Caries](resources\calculus.jpg)



# INTERFACES
1. # WEB INTERFACE
Ensure that you have all the requirements installed, especially the streamlit package to access the web interface locally
```bash
pip install -r requirements.txt
```
or 
```bash
pip install streamlit
```

To run the web interface use the commandline. 
```bash
streamlit run web/main.py
```
Make sure you are the the home directory. 
```bash
cd SW-AI-51 Video-Embedding–Powered Large Vision Models for Dental Disease Detection in Uganda
```
