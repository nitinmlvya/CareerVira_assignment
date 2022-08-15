import argparse
from pathlib import Path
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, Dropout, \
    Flatten, Dense
from sklearn.model_selection import train_test_split
from generate_data import normalize, to_categorical, do_data_augmentation
from tensorflow.keras import Sequential
import tensorflow as tf

SEED = 25

tf.random.set_seed(SEED)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# ===  Read image and make data and label sets === #
data = []
classes = []
data_dir = Path(args['data'])
for dir_path in data_dir.iterdir():
    label = dir_path.name
    for file_path in dir_path.iterdir():
        image = cv2.imread(str(file_path))
        image = cv2.resize(image, (32, 32))
        data.append(image)
        classes.append(label)

# === Normalize image data === #
data = normalize(data)

# === one-hot encoding over classes === #
classes, class_names = to_categorical(classes)
print('class_names: ', class_names)

# === Train test split === #
X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.20, random_state=SEED)

# === Augment data=== #
datagen = do_data_augmentation(X_train)

# === callbacks === #
checkpoint = ModelCheckpoint('models/model111-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')


# === Construct model === #

# initialize the model along with the input shape to be
# "channels last" and the channels dimension itself
model = Sequential()
input_shape = (32, 32, 3)
channel_dim = -1

# first CONV => RELU => CONV => RELU => POOL layer set
model.add(Conv2D(16, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# second CONV => RELU => CONV => RELU => POOL layer set
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# softmax classifier
model.add(Dense(classes.shape[1]))
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
print(model.summary())

# === Start training === #
BATCH_SIZE = 16
EPOCHS = 100
model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // BATCH_SIZE, epochs=EPOCHS, callbacks=[checkpoint])

# === Evaluate model === #
y_pred = model.predict(X_test, batch_size=16)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=class_names))