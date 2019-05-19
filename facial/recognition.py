import os
import cv2 as cv
from facial.helpers import vision, training, config


classifier = training.load('model.sav')
label_encoder = training.load('label.sav')
face_cascade = vision.load_cascade('haarcascade_frontalface_alt2.xml')
camera = cv.VideoCapture(0)
hog = vision.init_hog()


def do_prediction(roi):
    to_predict = cv.resize(roi, (50, 50), interpolation=cv.INTER_AREA)
    to_predict_hog = hog.compute(to_predict).ravel()
    predicted = classifier.predict([to_predict_hog])
    label = label_encoder.inverse_transform(predicted)
    return label


while True:
    _, frame = camera.read()
    frame = cv.flip(frame, 1)
    for_picture = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        id_number = do_prediction(roi_gray)
        cv.putText(frame, id_number[0], (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv.imshow('Project Facial', frame)
    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == 27:
        break

camera.release()
cv.destroyAllWindows()
