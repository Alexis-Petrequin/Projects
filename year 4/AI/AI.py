import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('model')

img_size = (224, 224)

# Define the path to the folder containing the images
# folder_path = 'datasets/chest_Xray/train/NORMAL/'
# folder_path = 'datasets/test/'
# folder_path = 'datasets/chest_Xray/test/PNEUMONIA'
folder_path = 'datasets/keynote'
normal_count = 0
pneumonia_count = 0

# Iterate over all the images in the folder
for filename in os.listdir(folder_path):
    # Load the image and preprocess it
    img = load_img(os.path.join(folder_path, filename), target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make the prediction
    prediction = model.predict(img_array)


   # Print the predicted class
    if prediction < 0.5:
        print(f'{filename} is healthy.')
        normal_count += 1
    else:
        print(f'{filename} has pneumonia.')
        pneumonia_count += 1

# Print the final counts
print(f'Number of normal lungs: {normal_count}')
print(f'Number of pneumonia lungs: {pneumonia_count}')
