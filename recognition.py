import os
import cv2 as cv
from skimage.feature import hog
from sklearn.externals import joblib

classifier = joblib.load('trained.yml')
label_encoder = joblib.load('labels.yml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv.CascadeClassifier(os.path.join(cv.__path__[0], 'data', 'haarcascade_frontalface_alt2.xml'))
camera = cv.VideoCapture(0)


def do_prediction(roi):
    to_predict = cv.resize(roi, (50, 50), interpolation=cv.INTER_AREA)
    to_predict = hog(to_predict, pixels_per_cell=(10, 10), cells_per_block=(1, 1))
    prediction = classifier.predict_proba([to_predict])
    print(prediction[0][0])
    predicted = classifier.predict([to_predict])
    return label_encoder.inverse_transform(predicted)


while True:
    _, frame = camera.read()
    frame = cv.flip(frame, 1)
    for_picture = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (39, 35, 145), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        id_number = do_prediction(roi_gray)
        cv.putText(frame, id_number[0], (x, y + h), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv.imshow('Project Facial', frame)
    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == 27:
        break
camera.release()
cv.destroyAllWindows()