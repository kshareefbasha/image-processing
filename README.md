# image-processing
this repository is about image processing ai tool
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Upload the image to Google Colab
from google.colab import files
uploaded = files.upload()

# Get the file name of the uploaded image
image_filename = list(uploaded.keys())[0]

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

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Perform image classification with the uploaded image
classify_image(image_filename)
