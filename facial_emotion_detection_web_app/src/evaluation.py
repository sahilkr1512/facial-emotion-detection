"""
evaluation.py
Evaluates the trained model on the FER2013 PrivateTest split: confusion
matrix, per-class precision/recall/F1, and one-vs-rest ROC/AUC curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

DATA_DIR = 'data/processed'
MODEL_PATH = 'models/emotion_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    X_test = np.load(f'{DATA_DIR}/X_test.npy')
    y_test = np.load(f'{DATA_DIR}/y_test.npy')

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(classification_report(y_test, y_pred, target_names=EMOTIONS, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix - FER2013 PrivateTest')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()

    y_test_bin = label_binarize(y_test, classes=list(range(7)))
    plt.figure(figsize=(8, 7))
    for i in range(7):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{EMOTIONS[i]} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - One-vs-Rest')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=150)
    plt.close()

    print("Saved confusion_matrix.png and roc_curves.png to outputs/")


if __name__ == '__main__':
    main()
