# IMPORTS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.callbacks import ModelCheckpoint


# print(list(os.walk('../input/')))
#CONSTANTS
NO_OF_CLASSES = 29
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
BATCH_SIZE = 50
DATASET_PATH = '../input/asl_alphabet_train/asl_alphabet_train'
EPOCHS = 5

def createClassifier():
    classifier = Sequential()
    classifier.add(Conv2D(16 , (2,2) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    #classifier.add(Conv2D(16 , (2,2) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2 , 2) , strides=(2,2), padding='same'))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(32 , (5,5) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    #classifier.add(Conv2D(32 , (5,5) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(5 ,5) , strides=(5,5), padding='same'))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(64 , (5,5) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    #classifier.add(Conv2D(64 , (5,5) , input_shape = (IMAGE_HEIGHT , IMAGE_WIDTH , 3) , activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(5 , 5) , strides=(5,5), padding='same'))
    classifier.add(Dropout(0.25))

    classifier.add(Flatten())

    classifier.add(Dense(units=128 , activation = 'relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=128 , activation = 'relu'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(units=NO_OF_CLASSES , activation='softmax'))

    #classifier.compile(optimizer = optimizers.SGD(lr=1e-2) , loss='categorical_crossentropy',metrics=['accuracy'])
    classifier.compile(optimizer = 'adam' , loss='categorical_crossentropy',metrics=['accuracy'])


    filepath="checkpoint-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return classifier


def get_training_and_test_set():
    generator = ImageDataGenerator(rescale = 1./255 , 
                         shear_range = 0.2 , 
                         zoom_range = 0.2 , 
                         horizontal_flip = True,
                         validation_split=0.2)

    training_set = generator.flow_from_directory(DATASET_PATH , 
                                    target_size = (IMAGE_HEIGHT , IMAGE_WIDTH) , 
                                    batch_size = BATCH_SIZE,
                                    class_mode='categorical',
                                    subset='training')

    testing_set = generator.flow_from_directory(DATASET_PATH , 
                                    target_size = (IMAGE_HEIGHT , IMAGE_WIDTH) , 
                                    batch_size = BATCH_SIZE,
                                    class_mode='categorical',
                                    subset='validation')
    return training_set , testing_set

def showGraphicalVisualisation(history):
    N = EPOCHS
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Digits Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

classifier = createClassifier()

training_set , testing_set = get_training_and_test_set()

history = classifier.fit_generator(training_set , 
                               steps_per_epoch = training_set.samples// BATCH_SIZE, 
                               epochs = EPOCHS ,
                               validation_data=testing_set, 
                               validation_steps=testing_set.samples// BATCH_SIZE)

classifier.save("check1.h5")
# showGraphicalVisualisation(history)