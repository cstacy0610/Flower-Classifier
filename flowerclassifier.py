


#CSC 375
#Machine Learning
#Semester Project
#Chris Stacy
#Flower Image Classifier

#Library imports
import numpy
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import os
import cv2

#Directory where images are located
images = 'C:\\icons\\school\\375 project\\flowers\\flower_data'

#Read, crop, and label flowers
flowers = []
labels = []

for fldr in os.listdir(images):
    for file in os.listdir(os.path.join(images, fldr)):
        labels.append(fldr)
        flower = cv2.imread(os.path.join(images, fldr, file))
        flower_to_rgbs = cv2.cvtColor(flower, cv2.COLOR_BGR2RGB)
        cropped_flower = cv2.resize(flower_to_rgbs, (64,64))
        flowers.append(cropped_flower)
        
flowers_nums = numpy.array(flowers)
labels_nums = numpy.array(labels)

enc = LabelEncoder() 
y = enc.fit_transform(labels_nums)
y = to_categorical(y, 102)
x = flowers_nums/255

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Show a prepared image
plt.imshow(x_train[0])

#Create CNN
#source used: https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
#source used: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

model = Sequential()                                #Input layer
model.add(Conv2D(32, (5,5), activation ='relu', input_shape = (64,64,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (5,5), activation='relu'))     #Hidden layer
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(102, activation = "softmax"))       #Output Layer

#Data augmentation
flower_randomizer = ImageDataGenerator(
        rotation_range=20,
        zoom_range = 0.20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True)

flower_randomizer.fit(x_train)

#Compile model
model.compile(optimizer = Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['acc'])
batch_size = 16
epochs = 64
hist = model.fit_generator(flower_randomizer.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, validation_data = (x_test, y_test), verbose = 1)

"""
#Model visualization - work in progress
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Visualization')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'lower right')
plt.show()
"""