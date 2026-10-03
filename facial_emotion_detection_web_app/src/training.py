import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from model import build_model

DATA_DIR = "data/processed"
MODEL_OUT = "models/emotion_model.h5"


def main():

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")

    X_val = np.load(f"{DATA_DIR}/X_val.npy")
    y_val = np.load(f"{DATA_DIR}/y_val.npy")

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)

    y_train_cat = tf.keras.utils.to_categorical(y_train, 7)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 7)

    model = build_model()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True
    )

    cbs = [
        callbacks.ModelCheckpoint(
            MODEL_OUT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(
        datagen.flow(
            X_train,
            y_train_cat,
            batch_size=64,
            shuffle=True
        ),
        validation_data=(X_val, y_val_cat),
        epochs=40,
        callbacks=cbs,
        verbose=1
    )

    model.save(MODEL_OUT)

    print("Training completed.")
    print("Model saved to:", MODEL_OUT)


if __name__ == "__main__":
    main()