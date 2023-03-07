import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # loop over all subdirectories
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.startswith('.'):
                # Read in the image file at the path os.path.join(root, file)
                try:
                    image = cv2.imread(os.path.join(root, file))
                    # Resize the image to the desired size
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    # Append the image and its corresponding label to the output lists
                    images.append(image)
                    labels.append(int(os.path.basename(root)))
                except Exception as e:
                    print(f"Error processing image {os.path.join(root, file)}: {e}")
                    
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # define the model architecture
    model = tf.keras.Sequential([
        # Convolutional layer with 32 filters of size 3x3, using ReLU activation function
        # Expects input image of shape (IMG_WIDTH, IMG_HEIGHT, 3)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        # Max pooling layer with pool size of 2x2 pixels
        # Reduces the spatial size of the representation
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Another convolutional layer identical to the first one
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        
        # Flattens the output of the previous layer into a 1D array
        tf.keras.layers.Flatten(),
        
        # Fully connected layer with 128 units, using ReLU activation function
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Dropout layer that randomly drops out 50% of the neurons during training to prevent overfitting
        tf.keras.layers.Dropout(0.5),
        
        # Output layer with NUM_CATEGORIES units, using softmax activation function
        # Produces a probability distribution over the output classes
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # compile the model with appropriate loss, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    main()
