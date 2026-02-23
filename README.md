# Real-Time Emotion Detection using CNN and Keras

This project implements a Convolutional Neural Network (CNN) model to detect human emotions from facial expressions. The model is trained on a large dataset of labeled facial images and deployed to perform real-time emotion recognition using webcam video input.

## Objective

To build a deep learning system capable of detecting and classifying human emotions from facial expressions in real-time.

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Haar Cascade Classifier (Face Detection)

## Model Architecture

- Convolutional Neural Network (CNN)
- Convolution Layers
- MaxPooling Layers
- Dropout for regularization
- Fully Connected Dense Layers
- Softmax activation for multi-class classification

## Dataset

FER-2013 Facial Expression Dataset  
(Contains grayscale facial images categorized into seven emotion classes)

## Features

- Face detection using OpenCV Haar Cascades
- Emotion classification using trained CNN model
- Real-time prediction via webcam video feed
- On-screen emotion display

## How to Run

1. Clone the repository
2. Install required libraries:
   pip install -r requirements.txt
3. Run:
   python real_time_emotion.py

## Applications

- Human-computer interaction
- Mental health monitoring
- Smart surveillance systems
- Customer sentiment analysis

## Author
Sameek Bhoir

Sameek  
MSc Data Science
