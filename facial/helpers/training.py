import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from . import config


def dump(data, filename):
    joblib.dump(data, os.path.join(config.PROJECT_DIR, 'training_result', filename))


def load(filename):
    return joblib.load(os.path.join(config.PROJECT_DIR, 'training_result', filename))
