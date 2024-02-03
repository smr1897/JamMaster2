import json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

TRAIN_DATASET_PATH = "train_dataset.json"
TEST_DATASET_PATH = "test_dataset.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)

    X = np.array(data["chromagram"])
    Y = np.array(data["labels"])
    return X,Y

def prepare_dataset(validation_size):
    X,Y = load_data(TRAIN_DATASET_PATH)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=validation_size)
    X_train = X_train[...,np.newaxis]
    X_validation = X_validation[...,np.newaxis]

    return X_train,X_validation,Y_train,Y_validation

def build_model(input_shape):
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(24,activation='softmax'))

    return model

if __name__ == "__main__":
    X_train,X_validation,Y_Train,Y_validation = prepare_dataset(0.2)
    x_test,y_test = load_data(TEST_DATASET_PATH)
    X_test = x_test[...,np.newaxis]
    Y_test = y_test

    input_shape = (X_train.shape[1],X_train.shape[2],1)
    model = build_model(input_shape)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(X_train,Y_Train,validation_data=(X_validation,Y_validation),batch_size=32,epochs=30)

    test_error,test_accuracy = model.evaluate(X_test,Y_test,verbose=1)
    print("Accuracy on test set : {}".format(test_accuracy))



