import os
import cv2 as cv
from sklearn.externals import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

classifier = joblib.load(os.path.join(BASE_DIR, 'training_result', 'model.sav'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'training_result', 'label.sav'))

face_cascade = cv.CascadeClassifier(os.path.join(cv.__path__[0], 'data', 'haarcascade_frontalface_alt2.xml'))
camera = cv.VideoCapture(0)

winSize = (50, 50)
blockSize = (10, 10)
blockStride = (5, 5)
cellSize = (5, 5)
nbins = 9
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


def do_prediction(roi):
    to_predict = cv.resize(roi, (50, 50), interpolation=cv.INTER_AREA)
    to_predict_hog = hog.compute(to_predict).ravel()
    prediction = classifier.predict_proba([to_predict_hog])
    predicted = classifier.predict([to_predict_hog])
    label = label_encoder.inverse_transform(predicted)
    print('{}: {:.2f}'.format(label[0], prediction[0][0]*100))
    return label


while True:
    _, frame = camera.read()
    frame = cv.flip(frame, 1)
    for_picture = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv.rectangle(frame, (x, y), (x + w, y + h), (39, 35, 145), 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        id_number = do_prediction(roi_gray)
        cv.putText(frame, id_number[0], (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv.imshow('Project Facial', frame)
    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == 27:
        break
camera.release()
cv.destroyAllWindows()