import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, AveragePooling2D
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
import gc
import pickle

x = []
y = []

# Define function for loadinh up all data for the images
def train_data_gen(DIR, ID):
    for img in os.listdir(DIR)[:750]:
        try:
            path = DIR + '/' + img
            img = cv2.imread(path)
            img = cv2.resize(img, (150, 150))
            if img.shape == (150, 150, 3):
                x.append(img)
                y.append(ID)
        except:
            None
# Creates nice progress bar when reading images
for DIR in tqdm(os.listdir('../input/plantvillage-dataset/color/')):
    train_data_gen('../input/plantvillage-dataset/color/' + DIR, DIR)

# Encode labers and set vars
print('reached label encoder')
le = LabelEncoder()
y = le.fit_transform(y)
del le
gc.collect()
x = np.array(x)
y = to_categorical(y, 38)


x_train,x_val,y_train,y_val = tts(x, y, test_size = 0.30, shuffle=True)

# Delete huge variables for memory management
del x
del y
gc.collect()

print('datagen')
# Data generator to create more images
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    zoom_range = 0.1,
    shear_range=0.1,
    fill_mode = "reflect",
    vertical_flip=True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
)


# Create generator for goal variables
valgen = ImageDataGenerator(
    rescale=1.0/255.0
)

gc.collect()

print('model')
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(150, 150, 3), padding='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(Dropout(0.27))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Dropout(0.27))
model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(38, activation='softmax'))


print('Model compile')
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, nesterov=True), metrics=['accuracy'])

print('Model fit')
model.fit_generator(datagen.flow(x_train,y_train,batch_size=8, shuffle=True), epochs=30, shuffle=True, steps_per_epoch=x_train.shape[0]//8, validation_data=valgen.flow(x_val,y_val,batch_size=8, shuffle=True), verbose=2)

# After training, save the model in a Keras model format. (HDF5)
model.save('plantus_model.h5')

