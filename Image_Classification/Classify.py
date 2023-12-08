#Importing Image Classification Libraries for Deeplearning

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#Loading the dataset
(x_train,y_train),(x_test,y_test) = cifar10.load_data()


'''
Converting the data type of pixel values in both the training 
and test sets to 32-bit floating-point numbers and normalizing 
the pixel values to the range [0, 1]. 

Crucial for improving the performance and
convergence of the neural network during training.
'''
x_train =x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255.0


'''
 Converting the categorical class labels 
 into a binary matrix format. Each row in the 
 matrix corresponds to a sample, and each column 
 corresponds to a class.
'''

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


'''
train_test_split: A function  to split datasets into training and testing sets.
By splitting the data into training and validation sets, a separate subset of 
data to evaluate the model's performance during training is ensured
and potential overfitting issues is avoided. 
'''

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

'''
Importing the libraries to build the deep learning
Model. These libraries below
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

print("Importing for deep learning model")

# Define the CNN model

model = Sequential()

'''
A convolutional neural network (CNN) with three convolutional layers, 
each followed by a max pooling layer. The convolutional layers learn 
hierarchical features from the input images, and the max pooling layers
reduce the spatial dimensions of the feature maps. 
'''

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

'''
Add a flattening layer to prepare the output for fully connected layers,
dense layer with ReLU activation, a dropout layer for regularization, and
a final dense layer with softmax activation for the classification task. 
'''

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

'''
Setting up the Adam optimizer with a custom initial learning rate 
and compiles the model with the specified optimizer, loss function, 
and metrics. Additionally, it defines a learning rate schedule function 
and creates a LearningRateScheduler object that can be used during model 
training to adaptively adjust the learning rate based on the epoch.
'''

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def lr_schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0005
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_val, y_val), callbacks=[lr_scheduler])

'''
The trained model is evaluated on the test set, print the 
test accuracy, and save the trained model to a file for future
use or deployment. The saved model can be loaded later using 
appropriate functions to make predictions on new data.
'''

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

model.save('cifar10_model.h5')

'''
Creating a figure with two subplots side by side. 
It assesses how well the model is learning from the 
training data and how well it generalizes to unseen validation data.
'''

# Plot training history
plt.figure(figsize=(12, 4))


'''
The first subplot shows the training and validation accuracy over epochs
'''

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

'''
The second subplot shows the training and validation loss over epochs.
'''

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()