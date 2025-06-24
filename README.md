# Emotion Detection with Smooth Grad-CAM++

📂 Project Overview

This project focuses on facial emotion detection using a Convolutional Neural Network (CNN) trained on grayscale facial images (48x48 pixels). It also implements Smooth Grad-CAM++ to visualize which regions of the image contribute to the model’s decision, providing explainability for the model's predictions.


🚀 Project Structure

emotion_detection.ipynb

➜ Jupyter Notebook for model training and evaluation.

grad_cam.py

➜ Python script for loading the trained model, making predictions, and visualizing activation maps using Smooth Grad-CAM++.


✅ Key Features

Emotion classification using CNN

Smooth Grad-CAM++ visualizations to interpret model predictions

Automatic visualization for all convolutional layers

Easy to adapt for different datasets or models

📂 Dataset

Dataset Source:
https://www.kaggle.com/datasets/msambare/fer2013

**Instructions:**
1. Download the FER-2013 dataset from the link above.
2. Place the downloaded dataset (fer2013.csv) in your project folder.
3. If required, convert the CSV to image files for direct image-based Grad-CAM visualization.
4. Run the `grad_cam.py` script to generate Grad-CAM visualizations.

> 💡 Note: Ensure the image paths in the `grad_cam.py` file match your dataset location.


Model file:
https://drive.google.com/file/d/1zEIzMiy_cgVSjboGjRIFaVmXfZRTaSjo/view?usp=sharing

Model Weights:
https://drive.google.com/file/d/1vuer3LheGJGj7dlHsJuJwtvhdah9lyoN/view?usp=sharing

 
