import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalize(data):
    return np.array(data, dtype="float") / 255.0

def to_categorical(labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return tf.keras.utils.to_categorical(labels, num_classes=len(set(labels))), le.classes_

def do_data_augmentation(data):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(data)
    return datagen
