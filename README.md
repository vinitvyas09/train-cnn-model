This Convolutional Neural Network (CNN) model is designed for emotion classification on human faces. The architecture includes multiple convolutional layers followed by fully connected layers, trained from scratch on a Kaggle dataset containing 35,685 examples of 48x48 pixel grayscale images of faces. The model classifies seven emotions: angry, disgust, fear, happy, sad, neutral, and surprise. Training was conducted with accuracy and loss metrics monitored, resulting in a robust model suitable for various applications such as sentiment analysis and emotion detection systems.

# Model
If you want to use this model directly without running this code, or are interested in the output of this training, you can download the weights from my kaggle repo here: https://www.kaggle.com/models/vinitvyas09/cnn_model_emotion_classification. Note that this model has a size of around 31MB.

# Usage
The model can be used for emotion classification by loading the pre-trained weights and performing inference on new face images. Below is a code snippet demonstrating how to load and use the model:

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('path_to_your_model.h5')

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion = np.argmax(prediction)
    return emotion

# Example usage
image_path = 'path_to_image.jpg'
emotion = predict_emotion(image_path)
print(f'The predicted emotion is: {emotion}')
Inputs are grayscale images of 48x48 pixels. Outputs are one of the seven emotion classes.

```

# System
This is a standalone model designed to classify emotions based on input images. It requires images to be preprocessed into 48x48 pixel grayscale format. The model can be integrated into larger systems for emotion detection and analysis.

# Implementation requirements
The model was trained using NVIDIA Jetson AGX boards, with TensorFlow and Keras libraries. Training required approximately 20 minutes on this hardware configuration. For inference, a GPU is recommended for optimal performance, though CPU can also be used with longer latency.

# Model Characteristics
## Model initialization
The model was trained from scratch using the Kaggle facial emotion recognition dataset: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/

## Model stats
### Model size
Approximately 30MB
### Number of layers
8 convolutional layers, 3 fully connected layers
### Latency
Approximately 20ms per image on GPU, 100ms on CPU
### Other details
The model is neither pruned nor quantized. Differential privacy techniques were not applied.

# Data Overview
## Training data
The dataset contains 35,685 grayscale images of faces, each 48x48 pixels, labeled with one of seven emotions. The data was preprocessed by normalizing pixel values to the range [0, 1] and augmenting the data through rotations and flips.

## Demographic groups
The dataset includes diverse demographic groups, although specific demographic attributes are not provided.

# Evaluation data
The data was split into training (80%), validation (10%), and test (10%) sets. No significant differences were observed between training and test data distributions.

# Usage limitations
The model's performance may degrade with images significantly different from the training data, such as those with extreme lighting conditions or occlusions. Users should ensure that input images are preprocessed appropriately to match the training conditions.

# Ethics
Ethical considerations included ensuring the model does not reinforce harmful stereotypes or biases. The model was evaluated for fairness, and steps were taken to balance the training data. However, users should be cautious of potential biases and continually monitor and assess the model's impact in real-world applications.
