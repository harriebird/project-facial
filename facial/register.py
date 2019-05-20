import os
import cv2 as cv
import time
from helpers import vision, config

face_cascade = vision.load_cascade('haarcascade_frontalface_alt2.xml')
eye_cascade = vision.load_cascade('haarcascade_eye.xml')
camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()
    frame = cv.flip(frame, 1)
    for_picture = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    cv.imshow('Register your beautiful face', frame)
    pressed_key = cv.waitKey(1) & 0xFF
    if pressed_key == 27:
        break
    elif pressed_key == 32:
        file_name = 'img_{}.jpg'.format(time.strftime('%Y%m%d_%H%M%S'))
        ans = str(input('Do you want to save this? y/[n]: '))
        if ans.lower() == 'y':
            name = str(input('Enter your ID number:'))
            file_path = os.path.join(config.PROJECT_DIR, 'images', name, file_name)
            if not os.path.exists(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
            cv.imwrite(file_path, for_picture)
        ans = str(input('Do you want to add new entry? y/[n]: '))
        if ans.lower() == 'y':
            continue
        else:
            break

camera.release()
cv.destroyAllWindows()
