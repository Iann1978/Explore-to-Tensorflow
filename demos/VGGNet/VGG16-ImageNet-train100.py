# Import keras and other Utils
import pltutils as pltu
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os


from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D
from tensorflow.keras.layers import Reshape, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, InputLayer, Lambda, Dropout
from tensorflow.keras.layers import BatchNormalization,Softmax
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau


%matplotlib inline
plt.style.use('seaborn-whitegrid')


# Create Model
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(224,224,3),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D((2,2)))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D((2,2)))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
model.add(Conv2D(512,(3,3),padding='same',activation='relu'))

model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dense(100,activation='softmax'))
print (model.summary())
model.save('checkpoints/vgg16-train100/model.h5')

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=1e-2, decay=5e-4, momentum=0.9, nesterov=True),
              metrics=['accuracy'])




# Callback for ModelCheckpoint
checkpoint_path = "checkpoints/vgg16-train100/checkpoint-weights_only-epoch{epoch:04d}-accuracy{accuracy:.4f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
cp_callback=ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True,period=1)

# Callback for TensorBoard
tb_callback = TensorBoard(log_dir='e:\\logs\\')

# Callback for ReduceLROnPlateau
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                verbose=0, mode='auto', cooldown=0, min_lr=0)

# Create and load ImageDataGenerator
batch_size = 32
datagen=ImageDataGenerator()   
 
train_generator = datagen.flow_from_directory(
    'e:/tensorflow/data/ImageNet/train100',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size)

validation_generator = datagen.flow_from_directory(
    'e:/tensorflow/data/ImageNet/val100',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=batch_size)

# Train
model.fit_generator(generator=train_generator,steps_per_epoch=129395/batch_size,epochs=50
                   ,callbacks=[tb_callback,tb_callback, lr_callback],
                   validation_data=validation_generator,validation_steps=batch_size)

# Eluvate
model.evaluate_generator(generator=validation_generator)


#model.load_weights('./checkpoints/vgg16-train10/checkpoint-weights_only-epoch0016-accuracy0.8942.ckpt')








