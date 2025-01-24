# Leaf-Disease-Detection

This repository focuses on detecting diseases in plant leaves using deep learning techniques. The project involves dataset preparation, preprocessing, model training, evaluation, and visualization of results.

# Features

Preprocessing leaf images for disease classification.
Building and training convolutional neural networks (CNNs) for image classification.
Evaluating model performance using metrics such as accuracy and F1-score.
Visualizing predictions and performance metrics.

Dependencies
Ensure the following Python libraries are installed:

tensorflow
keras
numpy
matplotlib
seaborn
opencv-python
scikit-learn

# Dataset

The dataset contains images of plant leaves, categorized into healthy and diseased classes. Each image represents either a healthy leaf or a leaf affected by a specific disease.

Dataset Structure

Training set: Images used to train the model.
Validation set: Images used to tune the model.
Test set: Images used to evaluate the model's performance.

Data Preprocessing
Resize images to a uniform dimension.
Normalize pixel values to a range of [0, 1].
Augment data using techniques like rotation, flipping, and zoom to improve generalization.

# Code Overview

1. Data Loading and Preprocessing
Load dataset from the specified directory.
Apply transformations like resizing and normalization.
Split data into training, validation, and test sets.

2. Model Architecture
Build a Convolutional Neural Network (CNN) using keras.Sequential.
Include layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
Use ReLU activation for hidden layers and softmax for the output layer.

3. Model Training
Compile the model with:
Loss: categorical_crossentropy
Optimizer: adam
Metrics: accuracy
Train the model with callbacks like ModelCheckpoint and EarlyStopping.

4. Model Evaluation

Evaluate the model on the test set.
Compute metrics like accuracy, precision, recall, and F1-score.
Plot confusion matrix to analyze classification performance.

5. Visualization

Plot training and validation accuracy/loss over epochs.
Visualize predictions on sample images, highlighting correct and incorrect predictions.
The trained model achieved the following performance on the test set:

Accuracy: [Insert accuracy value].
Precision: [Insert precision value].
Recall: [Insert recall value].

F1-Score: [Insert F1-score value].
