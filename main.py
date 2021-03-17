import os
import numpy as np
import cv2
import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()


def predictTestAndSave(classifier):
    print("Saving..")

    test_path = 'test/'
    test_name = 'submission'

    test_images_names = [f for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    tst_images = []

    for tst_image in test_images_names:
        tst_images.append(cv2.imread(test_path + tst_image, cv2.IMREAD_GRAYSCALE))

    test_images = np.array(tst_images)
    test_images = test_images / 255.0

    test_prediction = classifier.predict(test_images)

    tst_file = open("submissions/" + test_name + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv', 'w')
    tst_file.write("id,label\n")

    for i in range(0, len(tst_images)):
        tst_file.write(str(test_images_names[i]) + "," + str(np.argmax(test_prediction[i])) + "\n")

    tst_file.close()
    print("Saved!")


IMG_SIZE = 32
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("Loading images..")

train_labels = np.loadtxt('train_true_labels.txt', 'int').astype(int)
validation_labels = np.loadtxt('validation_true_labels.txt', 'int').astype(int)

train_path = 'train/'
validation_path = 'validation/'

train_images_names = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
validation_images_names = [f for f in os.listdir(validation_path) if os.path.isfile(os.path.join(validation_path, f))]


t_images, v_images = [], []

for t_image in train_images_names:
    t_images.append(cv2.imread(train_path + t_image, cv2.IMREAD_GRAYSCALE))

for v_image in validation_images_names:
    v_images.append(cv2.imread(validation_path + v_image, cv2.IMREAD_GRAYSCALE))

# train_images = np.array(t_images).reshape(-1, IMG_SIZE * IMG_SIZE)
# validation_images = np.array(v_images).reshape(-1, IMG_SIZE * IMG_SIZE)

train_images = np.array(t_images)
validation_images = np.array(v_images)

train_images = train_images / 255.0
validation_images = validation_images / 255.0

print('Images loaded!')

# print("Scaling!")
# scaler = StandardScaler()
# scaler.fit(train_images)
# train_images_scaled = scaler.transform(train_images)


print("Creating model!")
# classifier = SVC(C=1, kernel='poly', gamma='auto')
classifier = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(9, activation=tf.nn.softmax)
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Started training!")

classifier.fit(train_images, train_labels, epochs=30)

print("Predicting!")
prediction = classifier.predict(validation_images)

# score_mean = accuracy_score(validation_labels, prediction)
validation_loss, validation_acc = classifier.evaluate(validation_images, validation_labels)

# print("Prediction score by mean: " + str(score_mean * 100) + "%")
print("Prediction score: " + str(validation_acc * 100) + "%")

# Save to file
predictTestAndSave(classifier)
