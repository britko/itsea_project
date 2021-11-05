import cv2
import numpy as np
import matplotlib.pyplot as plt



video_path = r'C:\Users\pc\Downloads\itsea progect\make\ship.mp4'
cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

tracker=cv2.TrackerMIL_create()
ret,img=cap.read()

cv2.namedWindow('select Window')
rect =cv2.selectROI('Select Windo',img,fromCenter=False,showCrosshair=True)
cv2.destroyWindow('Select Windo')

tracker.init(img,rect)



success, box=tracker.update(img)
left,top,w,h=[int(v) for v in box]


src = np.float32([[0, 700], [1280, 700], [0, 0], [1280, 0]])
dst = np.float32([[569, 700], [711, 700], [0, 0], [1280, 0]])
M = cv2.getPerspectiveTransform(src, dst)    # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


#동영상 받아왔는지 확인
while True:
    ret, img=cap.read()
    img = img[top:(top + h), left:left + w]
    if not ret:
        exit()

    warped_img = cv2.warpPerspective(img, M, (1280, 700))  # Image warping


    cv2.imshow('img',warped_img)
    if cv2.waitKey(1) ==ord('q'):
        break









"""이미지 위에서 보기
IMAGE_H = 223
IMAGE_W = 1280

src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)    # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


img = cv2.imread('test2_img.jpg',cv2.IMREAD_COLOR)



# Read the test img
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()


      # Apply np slicing for ROI crop
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()

warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()


img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, IMAGE_H)) # Inverse transformation
plt.imshow(cv2.cvtColor(img_inv, cv2.COLOR_BGR2RGB))      # Show results
plt.show()

"""



