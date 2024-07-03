import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import os

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# Initialize and compile the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model_path = 'mnist_model.h5'
model.save(model_path)

# Print the absolute path of the saved model
abs_model_path = os.path.abspath(model_path)
print(f"Model saved to {abs_model_path}")
