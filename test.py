import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math 
import time

cap =cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

classifier = Classifier("Models/keras_model.h5","Models/labels.txt")
offset = 20
imgSize =300
counter =0
folder = "DATA"
labels =["A","B","C"]

while True:
    success , img =cap.read()
    if not success:
        break

    hands , img = detector.findHands(img)

    if hands :

        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite =np.ones((imgSize , imgSize ,3),np.uint8)*255

     
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

     
        if imgCrop.size == 0:
            continue


        # imgCropShape =imgCrop.shape

      
        aspectRatio =h/w

        if aspectRatio >1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop ,(wCal , imgSize))  
            # imgResizeShape = imgResize.shape
            wGap =math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)
            cv2.putText(img, labels[index], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop ,( imgSize, hCal)) 
            # imgResizeShape = imgResize.shape
            hGap =math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        
        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

    cv2.imshow("Image", img)
    
  
      

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
