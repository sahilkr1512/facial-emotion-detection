import numpy as np
import tensorflow as tf
from model import build_model

X = np.load("data/processed/X_train.npy")
y = np.load("data/processed/y_train.npy")

# Use only 1000 images for the quick test
X = X[:1000]
y = y[:1000]

# Convert labels to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=7)

model = build_model()

model.fit(
    X,
    y,
    batch_size=32,
    epochs=10,
    verbose=1
)