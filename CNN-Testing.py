import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Sequential 
from keras.models import model_from_json
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os

print(tf.__version__)

#  width is [0]
# height it [1]
x = []

# Classes for all the different plants and diseases the model can predict
CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# A thrown-together function that crops the image
def prep_pic(img_path):
    img = Image.open(img_path)
    old_width  = img.size[0] 
    old_height = img.size[1]
    
    square_width = 150
    square_height = 150
    
    if old_height > old_width:
        square_width = old_width
        square_height = old_width
    
    if old_height <= old_width:
        square_width = old_height
        square_height = old_height
    
    left = (old_width - square_width)/2
    top = (old_height - square_height)/2
    right = (old_width + square_width)/2
    bottom = (old_height + square_height)/2

    img2 = img.crop((left, top, right, bottom))
    img2.resize = (-1, 150, 150, 3)

    
    path = img_path
    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))
    x.append(img)

prep_pic('../input/project-plantus/corn.jpg')
x = np.array(x)
x = x / 255.0
model = Sequential()
model = load_model('../input/project-plantus/plantus_model.h5')
print(CLASSES[model.predict_classes([x])[0]-1], "or", CLASSES[model.predict_classes([x])[0]], '(', model.predict_classes([x]), ')')], CLASSES[16])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
