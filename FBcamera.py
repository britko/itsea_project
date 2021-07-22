import cv2
import numpy as np

cam_F = cv2.VideoCapture(0)
cam_F.set(3, 1920)
cam_F.set(4, 1080)

cam_B = cv2.VideoCapture(1)
cam_B.set(3, 1920)
cam_B.set(4, 1080)

# if cam.isOpen():
#     print('width: {}, height : {}'.format(cam.get(3), cam.get(4)))
	
while True:
    ret_F, fram_F = cam_F.read()

    ret_B, fram_B = cam_B.read()
	
    if ret_F & ret_B:

        # 전방 camera
        img_F = cv2.cvtColor(fram_F, cv2.COLOR_BGR2RGB)

        IMAGE_H = 300
        IMAGE_W = 1280

        src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
        Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

        img_F = img_F[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
        warped_img_F = cv2.warpPerspective(img_F, M, (IMAGE_W, IMAGE_H)) # Image warping


        # 후방 camera
        img_B = cv2.cvtColor(fram_B, cv2.COLOR_BGR2RGB)

        img_B = img_B[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
        warped_img_B = cv2.warpPerspective(img_B, M, (IMAGE_W, IMAGE_H)) # Image warping

        warped_img_B_flip = cv2.flip(warped_img_B, 0) # 1은 좌우 반전, 0은 상하 반전입니다.




        ## 결과 이미지 출력
        final_frame = cv2.vconcat([warped_img_F,warped_img_B_flip])
        # cv2.imshow('bird-eye-view1', cv2.cvtColor(warped_img_F, cv2.COLOR_BGR2RGB))
        # cv2.imshow('bird-eye-view2', cv2.cvtColor(warped_img_B_flip, cv2.COLOR_BGR2RGB))
        cv2.imshow('birds-eye-view', cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))

        # q 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
	    print('error')

cam_F.release()
cam_B.release()
cv2.destroyAllWindows()