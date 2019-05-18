import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv.CascadeClassifier(os.path.join(cv.__path__[0], 'data', 'haarcascade_frontalface_alt2.xml'))

winSize = (50, 50)
blockSize = (10, 10)
blockStride = (5, 5)
cellSize = (5, 5)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

training_data = []
training_labels = []


for root, dirs, files in os.walk(os.path.join(BASE_DIR, 'images')):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            file_path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()

            image = cv.imread(file_path)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_array = np.array(gray_image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            print(faces)
            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                roi = cv.resize(roi, (50, 50), interpolation=cv.INTER_AREA)
                hog_data = hog.compute(roi).ravel()
                training_data.append(hog_data)
                training_labels.append(label)

classifier = MLPClassifier()
label_encoder = LabelEncoder()
label_encoder.fit(training_labels)
training_data = np.array(training_data)

x_train, x_test, y_train, y_test = train_test_split(training_data,
                                                    label_encoder.transform(training_labels), test_size=0.5)

print(x_train[0])

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

joblib.dump(classifier, os.path.join(BASE_DIR, 'training_result', 'model.sav'))
joblib.dump(label_encoder, os.path.join(BASE_DIR, 'training_result', 'label.sav'))

print('Hell Yeah! Training was successfully done. :)')