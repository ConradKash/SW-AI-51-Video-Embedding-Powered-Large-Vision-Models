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
