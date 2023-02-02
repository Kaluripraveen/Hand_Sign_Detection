import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

#turning on your webcam
cap = cv2.VideoCapture(0)
#hand detector
detector = HandDetector(maxHands=1)
offset = 20
imgsize= 300
counter = 0

folder = "dataa/Y"
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        #asking values of bounding box
        x,y,w,h = hand['bbox']
        # creating the white box
        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        imageWhite = imgCrop.shape
        #overlapping the croppedimg image to whiteboximg 
        #imgWhite[0:imageWhite[0],0:imageWhite[1]] = imgCrop
        # placing overlaped image in center
        aspectratio = h/w
        
        if aspectratio> 1:
            k = imgsize/h
            wcal= math.ceil(k*w)
            imgResize= cv2.resize(imgCrop,(wcal,imgsize))
            imageResizeShape = imgResize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgWhite[:,wgap:wcal+wgap] = imgResize
            
        else:
            k = imgsize/w
            hcal= math.ceil(k*h)
            imgResize= cv2.resize(imgCrop,(imgsize,hcal))
            imageResizeShape = imgResize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgWhite[hgap:hcal+hgap,:] = imgResize
        
        
        cv2.imshow("imgCrop",imgCrop)
        cv2.imshow("imgWhite",imgWhite)
    cv2.imshow("image",img)
    key= cv2.waitKey(1)
    if key==ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)