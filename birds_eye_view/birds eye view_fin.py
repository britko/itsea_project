import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, fram = cam.read()

    if ret:

        img = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)

        IMAGE_H, IMAGE_W = img.shape[:2]

        src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

        img = img[0:IMAGE_H, 0:IMAGE_W] # Apply np slicing for ROI crop
        wraped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping

        crop_img = wraped_img[500:IMAGE_H, int(IMAGE_W/2-100):int(IMAGE_W/2+100)]

        resize_img = cv2.resize(crop_img, None,  None, 3, 3, cv2.INTER_CUBIC)

        # 결과 출력
        #cv2.imshow('img', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #cv2.imshow('wraped_img', cv2.cvtColor(wraped_img, cv2.COLOR_BGR2RGB))
        #cv2.imshow('crop_img', cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        cv2.imshow('bev', cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))

        # q 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
	    print('error')

cam.release()
cv2.destroyAllWindows()
