# Fruition-16-Deep-Learning-Framework-for-Fruits-Freshness-Classification

This project focuses on classifying different types of fruits using a deep learning model built with TensorFlow and Keras. The dataset includes images of various fruits, and the model is trained to differentiate between them. The project follows a structured approach, including data preparation, model training, evaluation, and model deployment for prediction.

## Table of Contents
1. Data Preparation
2. Model Architecture
3. Model Training
4. Model Evaluation
5. Model Testing
6. Confusion Matrix and Classification Report
7. Predicting New Images

## Data Preparation

The dataset used in this project consists of images of different types of fruits. The data is organized into folders, with each folder representing a class of fruit. The data is split into training, validation, and test sets.

- **Training Set**: Used to train the model.
- **Validation Set**: Used to validate the model during training to avoid overfitting.
- **Test Set**: Used to evaluate the model's performance on unseen data.

The `ImageDataGenerator` is used to preprocess and augment the images to increase dataset diversity and improve the model's generalization ability.

## Model Architecture

The model is built using the Keras Sequential API. It consists of convolutional layers for feature extraction and dense layers for classification.

- **Convolutional Layers**: These layers detect features in the images.
- **Pooling Layers**: These layers reduce the spatial dimensions.
- **Dense Layers**: These layers classify the images based on extracted features.

## Model Training

The model is compiled with categorical cross-entropy loss, the Adam optimizer, and accuracy as the evaluation metric. The model is trained using the training set, and its performance is monitored on the validation set. The training process involves monitoring accuracy and loss over a set number of epochs.

Training and validation accuracy and loss are plotted to visualize the model's performance over time.

![Training and Validation Accuracy](images/training_validation_accuracy.png)

![Training and Validation Loss](images/training_validation_loss.png)

## Model Evaluation

After training, the model is evaluated on the test set to determine its generalization performance. The accuracy on the test set indicates how well the model performs on unseen data.

## Model Testing

To test the model with a new image, the image is loaded and preprocessed before using the model to predict the class. This approach assesses the model's ability to classify new, unseen images.

![Test Image](images/test_image.png)

## Confusion Matrix and Classification Report

To gain insights into the model's performance, a confusion matrix and a classification report are generated. The confusion matrix shows how well the model classifies each class, while the classification report provides precision, recall, and F1-score for each class.

![Confusion Matrix](images/confusion_matrix.png)

The confusion matrix helps visualize the distribution of correct and incorrect predictions, while the classification report offers detailed metrics for each class, including precision, recall, and F1-score.

## Predicting New Images

The model can be used to predict new images after loading the trained model. This section demonstrates how to load a new image and predict its class using the trained model.

![Predicted Test Image](images/predicted_test_image.png)

## Conclusion

This project demonstrates the use of deep learning to classify fruit images. The trained model can classify images with high accuracy, as indicated by the test set performance. The confusion matrix and classification report provide deeper insights into the model's classification abilities. The project also shows how to use the trained model to predict new images, making it useful for real-world applications.
