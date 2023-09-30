# Image Processing AI Tool

This repository contains an image processing AI tool that classifies images using a pre-trained ResNet50 model.

## Usage

Follow these steps to use the image processing AI tool:

1. Upload an image to Google Colab by running the following code:

   ```python
   from google.colab import files
   uploaded = files.upload()
Get the file name of the uploaded image:
image_filename = list(uploaded.keys())[0]
Run the image classification code by executing the following:
!python image_classification.py
This code will provide the top 3 predicted classes for the uploaded image.
Requirements
TensorFlow 2.x
Keras
numpy
Environment Setup
Install the required libraries in your Google Colab environment:
!pip install tensorflow
Clone this repository to your Colab environment:
!git clone https://github.com/kshareefbasha/image-processing.git
Example Output
1: golden_retriever (0.87)
2: Labrador_retriever (0.05)
3: kuvasz (0.03)
License
This project is licensed under the MIT License - see the LICENSE.md file for details.

**Improved image_classification.py (Python script):**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Define a function for image classification
def classify_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Print the top 3 predicted classes
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {label} ({score:.2f})")

# Get the file name of the uploaded image
image_filename = input("Enter the path to the image file: ")

# Perform image classification with the uploaded image
classify_image(image_filename)




