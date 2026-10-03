"""
preprocessing.py
Load the real FER2013 dataset (download fer2013.csv from Kaggle and place
it in data/) and prepare train/val/test numpy arrays for training.
"""
import numpy as np
import pandas as pd
import os

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48
DATA_PATH = 'data/fer2013.csv'
OUT_DIR = 'data/processed'


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)

    print("Raw shape:", df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=['pixels']).reset_index(drop=True)

    # Keep only rows with a valid 48*48 pixel string
    valid = df['pixels'].apply(lambda s: len(str(s).split()) == IMG_SIZE * IMG_SIZE)
    df = df[valid].reset_index(drop=True)

    print("Clean shape:", df.shape)
    return df


def to_arrays(df):
    X = np.stack([
        np.array(p.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0
        for p in df['pixels']
    ])
    y = df['emotion'].values.astype(int)
    return X, y


def split_by_usage(df):
    """FER2013 ships its own Training / PublicTest / PrivateTest split."""
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']

    X_train, y_train = to_arrays(train_df)
    X_val, y_val = to_arrays(val_df)
    X_test, y_test = to_arrays(test_df)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    df = load_and_clean()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_usage(df)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f'{OUT_DIR}/X_train.npy', X_train)
    np.save(f'{OUT_DIR}/y_train.npy', y_train)
    np.save(f'{OUT_DIR}/X_val.npy', X_val)
    np.save(f'{OUT_DIR}/y_val.npy', y_val)
    np.save(f'{OUT_DIR}/X_test.npy', X_test)
    np.save(f'{OUT_DIR}/y_test.npy', y_test)
    print(f"Saved processed arrays to {OUT_DIR}/")


if __name__ == '__main__':
    main()
