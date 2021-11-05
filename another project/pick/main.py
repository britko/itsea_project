import cv2
import numpy as np

video_path = r'C:\Users\pc\Downloads\itsea progect\pick\test.mp4'
cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

tracker=cv2.TrackerMIL_create()
ret,img=cap.read()

cv2.namedWindow('select Window')
cv2.imshow('select Window',img)

rect =cv2.selectROI('Select Windo',img,fromCenter=False,showCrosshair=True)
cv2.destroyWindow('Select Windo')

tracker.init(img,rect)



#동영상 받아왔는지 확인
while True:
    ret, img=cap.read()

    if not ret:
        exit()

    success, box=tracker.update(img)
    left,top,w,h=[int(v) for v in box]

    cv2.rectangle(img,pt1=(left,top),pt2=(left+w,top+h),color=(255,255,255),thickness=3)

    cv2.imshow('img',img)
    if cv2.waitKey(1) ==ord('q'):
        break
