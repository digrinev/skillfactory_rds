import cv2
import numpy as np
# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        print('Unable to load camera. Use the command "xhost +"')
        pass
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow('img', gray)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()