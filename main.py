
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.initializers import HeNormal
from keras.layers import Dropout
from keras.optimizers import SGD

folderName = '.\\Dataset\\'
"""
In this assignment, I implemented two fully connected feed-forward neural networks: one with 2 layers and another with 3 layers.

The 3-layer network generally got worse accuracy. This is because more than 2 layers "overfits" the data.

More layers also made the model slightly slower to train and more prone to overfitting, which we addressed using Dropout.

Effects of the Number of Neurons in Hidden Layers
In the 2-layer model, I used 128 neurons in the hidden layer.
In the 3-layer model, I used 128 neurons in the first hidden layer and 64 neurons in the second.

Why Dropout and He Initialization?
I used Dropout (rate = 0.3) in hidden layers to prevent overfitting by randomly deactivating neurons during training.
I used He Initialization (he_normal) because it works well with ReLU activations, weights are initialized in a way that maintains stable gradients.

For this image classification task (on 32x32x3 RGB images with basic features), the 2-layer network is more effective, easier to train, and better at generalization.

More neurons or layers don’t guarantee better accuracy, especially when the dataset is not very large or complex.
The Dropout and He initialization worked as expected, helping mitigate overfitting and stabilize training.
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Cloudy:
    def __init__(self):
        self.imgName = "cloudy"
        self.classLabel = 0
        self.images, self.features = imageLoaderAndFeatureExtractor(range(1, 301), self.imgName)
        self.training_set = get_set(self.images, range(0, 150))#50% of images
        self.validation_set = get_set(self.images, range(150, 225))#25# of rest
        self.test_set = get_set(self.images, range(225, 300))#25# of rest
        print(bcolors.OKGREEN + "Cloudy initialized." + bcolors.ENDC)


class Shine:
    def __init__(self):
        self.imgName = "shine"
        self.classLabel = 1
        self.images, self.features = imageLoaderAndFeatureExtractor(range(1, 253), self.imgName)
        self.training_set = get_set(self.images, range(0, 126))#50% of images
        self.validation_set = get_set(self.images, range(126, 189))#25# of rest
        self.test_set = get_set(self.images, range(189, 251))#25# of rest
        print(bcolors.OKGREEN + "Shine initialized." + bcolors.ENDC)

class Sunrise:
    def __init__(self):
        self.imgName = "sunrise"
        self.classLabel = 2
        self.images, self.features = imageLoaderAndFeatureExtractor(range(1,357), self.imgName)
        self.training_set = get_set(self.images, range(0, 178))#50% of images
        self.validation_set = get_set(self.images, range(178, 267))#25# of rest
        self.test_set = get_set(self.images, range(267, 355))#25# of rest
        print(bcolors.OKGREEN + "Sunrise initialized." + bcolors.ENDC)


def get_set(images, rng):
    set = []
    for i in rng:
        set.append(images[i])

    return set

def setFormer():

    cloudy = Cloudy()
    shine = Shine()
    sunrise = Sunrise()

    training = []
    testing = []
    validation = []

    training.extend([(item, 0) for item in cloudy.training_set])
    training.extend([(item, 1) for item in shine.training_set])
    training.extend([(item, 2) for item in sunrise.training_set])

    testing.extend([(item, 0) for item in cloudy.test_set])
    testing.extend([(item, 1) for item in shine.training_set])
    testing.extend([(item, 2) for item in sunrise.training_set])

    validation.extend([(item, 0) for item in cloudy.validation_set])
    validation.extend([(item, 1) for item in shine.validation_set])
    validation.extend([(item, 2) for item in sunrise.validation_set])

    #Randomize training and testing data

    random.shuffle(training)
    random.shuffle(testing)
    random.shuffle(validation)

    return training, testing, validation

def imageLoaderAndFeatureExtractor(rng, imgName):
    images = []
    features = []
    for i in rng:
        img = cv2.imread(folderName + imgName + str(i) + ".jpg")
        if img is None:
            img = cv2.imread(folderName + imgName + str(i) + ".jpeg")
            if img is None:
                print("Image: " + imgName + str(i) + ".jpg" +" couldn't be loaded")
            else:
                print(bcolors.OKGREEN + "Image: " + imgName + str(i) + ".jpeg" + " settled and loaded" + bcolors.ENDC)
        else:
            resizedImg = cv2.resize(img, (32, 32))
            images.append(resizedImg)  # Add shine1,shine2... in array
            features.append(resizedImg.flatten())

    features = np.asarray(features, dtype=object)
    return images, features

#2 LAYERS
def feedForwardNeuralNetwork2Layer(training, testing):
    x_train = []
    y_train = []

    for train in training:
        x_train.append(np.array(train[0]).flatten() / 255.0)
        y_train.append(train[1])

    x_test = []
    y_test = []

    for test in testing:
        x_test.append(np.array(test[0]).flatten() / 255.0)
        y_test.append(test[1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(3072,), kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc}')

#3 LAYERS
def feedForwardNeuralNetwork3Layer(training, testing):
    x_train = []
    y_train = []

    for train in training:
        x_train.append(np.array(train[0]).flatten() / 255.0)
        y_train.append(train[1])

    x_test = []
    y_test = []

    for test in testing:
        x_test.append(np.array(test[0]).flatten() / 255.0)
        y_test.append(test[1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(3072,), kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    model.fit(x_train, y_train, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc}')

#2 LAYERS
def feedForwardNeuralNetwork2LayerValidation(validation):
    #Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    x_validation = []
    y_validation = []

    for valid in validation:
        x_validation.append(np.array(valid[0]).flatten() / 255.0)
        y_validation.append(valid[1])

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(3072,), kernel_initializer='he_normal'),
        Dropout(0.3),  # Regularization
        Dense(10, activation='softmax')
    ])

    #Batch Gradient Descent
    optimizer = SGD(learning_rate=0.01)  # You can change this
    model.compile(optimizer=optimizer,
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    #Train and Save Every 5 Epochs (manual loop for control)
    EPOCHS = 20
    loss_list = []
    acc_list = []

    for epoch in range(1, EPOCHS + 1):
        history = model.fit(
            x_validation, y_validation,
            batch_size=len(x_validation),  # Batch Gradient Descent
            epochs=1,
            verbose=0  # suppress per-step logs
        )

        loss = history.history['loss'][0]
        acc = history.history['sparse_categorical_accuracy'][0]
        loss_list.append(loss)
        acc_list.append(acc)

        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        if epoch % 5 == 0:
            model.save(f'model_epoch_{epoch}.keras')
            print(f"Model saved at epoch {epoch}")

    #Plot metrics
    plt.plot(range(1, EPOCHS + 1), loss_list, label='Loss')
    plt.plot(range(1, EPOCHS + 1), acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def feedForwardNeuralNetwork3LayerValidation(validation):
    #Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    x_validation = []
    y_validation = []

    for valid in validation:
        x_validation.append(np.array(valid[0]).flatten() / 255.0)
        y_validation.append(valid[1])

    x_validation = np.array(x_validation)
    y_validation = np.array(y_validation)

    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(3072,), kernel_initializer='he_normal'),
        Dropout(0.3),  # Regularization
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),  # Regularization
        Dense(10, activation='softmax')
    ])

    #Batch Gradient Descent
    optimizer = SGD(learning_rate=0.01)  # You can change this
    model.compile(optimizer=optimizer,
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    #Train and Save Every 5 Epochs (manual loop for control)
    EPOCHS = 20
    loss_list = []
    acc_list = []

    for epoch in range(1, EPOCHS + 1):
        history = model.fit(
            x_validation, y_validation,
            batch_size=len(x_validation),  # Batch Gradient Descent
            epochs=1,
            verbose=0  # suppress per-step logs
        )

        loss = history.history['loss'][0]
        acc = history.history['sparse_categorical_accuracy'][0]
        loss_list.append(loss)
        acc_list.append(acc)

        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

        if epoch % 5 == 0:
            model.save(f'model_epoch_{epoch}.keras')
            print(f"Model saved at epoch {epoch}")

    #Plot metrics
    plt.plot(range(1, EPOCHS + 1), loss_list, label='Loss')
    plt.plot(range(1, EPOCHS + 1), acc_list, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    training, testing, validation = setFormer()
    feedForwardNeuralNetwork2Layer(training, testing)
    feedForwardNeuralNetwork3Layer(training, testing)
    feedForwardNeuralNetwork2LayerValidation(validation)
    feedForwardNeuralNetwork3LayerValidation(validation)