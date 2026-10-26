# Facial Emotion Recognition

## 1. Project Overview

This project uses a Neural Network to classify facial images into seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The images are grayscale and have a size of 48 × 48 pixels.

## 2. Setup Steps

1. Open the notebook in Google Colab.
2. Upload and extract the dataset.
3. Import the required libraries.
4. Load and prepare the images.
5. Train and evaluate the model.
6. Run the Streamlit application to upload an image and predict its emotion.

## 3. Model Architecture

The model uses a Sequential Neural Network with the following layers:

- **Input Layer:** Receives 48 × 48 grayscale images.
- **Flatten Layer:** Converts each image into a one-dimensional array.
- **Dense Layer:** Contains 128 neurons with ReLU activation.
- **Output Layer:** Contains 7 neurons with Softmax activation to predict the emotion.

## 4. Model Performance

The model was evaluated using the test dataset.

- **Test Accuracy:** 38.39%
- **Test Loss:** 3.8

The results show that the model can classify facial emotions, but its accuracy is limited and some predictions may be incorrect.

## 5. Ethics and Limitations

- **Lighting:** Poor lighting may affect the prediction.
- **Image Quality:** Blurry images or different face angles may reduce accuracy.
- **Similar Emotions:** Some emotions may look similar, making classification difficult.
- **Facial Differences:** Differences in facial features may affect the results.
- **Ethical Considerations:** Facial expressions do not always show how a person truly feels. The predictions should not be treated as a person's actual emotions.

## 6. Conclusion

This project demonstrates how a Neural Network can classify facial images into seven emotion categories. The test results show that the model has limited accuracy, so its predictions are not always correct.
