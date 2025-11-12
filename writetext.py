import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import time
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier("Models/keras_model.h5", "Models/labels.txt")


offset = 20
imgSize = 300
labels = ["A", "B", "C"]


final_text = ""
hand_present = False
last_letter = ""
letter_start_time = 0
hold_time_required = 1.0     
same_letter_delay = 3.0         


def draw_text_box(img, text):
    overlay = img.copy()
    alpha = 0.6

  
    cv2.rectangle(overlay, (20, img.shape[0] - 100),
                  (img.shape[1] - 20, img.shape[0] - 30),
                  (0, 150, 255), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (20, img.shape[0] - 100),
                  (img.shape[1] - 20, img.shape[0] - 30),
                  (255, 255, 255), 2)
    
   
    cv2.putText(img, text, (40, img.shape[0] - 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


last_letter_time = 0

while True:
    success, img = cap.read()
    if not success:
        print(" Failed to capture image")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)

           
            if wGap + wCal > imgSize:
                wCal = imgSize - wGap
                imgResize = cv2.resize(imgResize, (wCal, imgSize))

            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)

            if hGap + hCal > imgSize:
                hCal = imgSize - hGap
                imgResize = cv2.resize(imgResize, (imgSize, hCal))

            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        letter = labels[index]

 
        current_time = time.time()


        if letter != last_letter:
            last_letter = letter
            letter_start_time = current_time
        else:
            elapsed = current_time - letter_start_time

            if elapsed >= hold_time_required and (current_time - last_letter_time) >= same_letter_delay:
                final_text += letter
                print(f"✅ Added Letter: {letter}")
                last_letter_time = current_time 
                letter_start_time = current_time

        hand_present = True

       
        cv2.putText(img, f"Letter: {letter}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        if hand_present:
            final_text += " "
            print(" Space added")
            hand_present = False
            last_letter = ""
            letter_start_time = 0

   
    draw_text_box(img, f"Output: {final_text}")

    cv2.imshow("Sign Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nFinal Output Text:", final_text)
