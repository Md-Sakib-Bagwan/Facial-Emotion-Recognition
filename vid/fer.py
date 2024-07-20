# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=3.0,
    horizontal_flip=True,
    fill_mode='nearest')

validation_data_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_data_gen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle= True
    )

validation_gen = validation_data_gen.flow_from_directory(
    validation_data_dir,
    color_mode= 'grayscale',
    target_size= (48,48),
    batch_size= 32,
    class_mode= 'categorical',
    shuffle= True
    ) 

class_labels = ['Angry','Disgust','Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_gen.__next__()

model =  Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#print(model.summary())

train_path = "data/train/"
test_path = "data/test/"

num_train_img = 0
for root, dirs, files in os.walk(train_path):
    num_train_img += len(files)
    
num_test_img = 0
for root, dirs, files in os.walk(test_path):
    num_test_img += len(files)

#print(num_train_img)
#print(num_test_img)

epoch = 11

history = model.fit(train_gen,
                    steps_per_epoch=num_train_img//32,
                    epochs=epoch,
                    validation_data=validation_gen,
                    validation_steps=num_test_img//32)

model.save("modelfile_11_ep.h5")

