import cv2
import numpy as np
import matplotlib.pyplot as plt

""" 사각형 만들기

def find_chars(contour_list):
    matched_result_idx=[]

    for d1 in contour_list:
        matched_contours_idx=[]
        for d2 in contour_list:
            if d1['idx']==d2['idx']:
                continue
            dx=abs(d1['cx']-d2['cx'])
            dy=abs(d1['cy']-d2['cy'])

            diagonal_length1=np.sqrt(d1['w']**2+d1['h']**2)

            distance=np.linalg.norm(np.array([d1['cx'],d1['cy']])-np.array([d2['cx'],d2['cy']]))
            if dx==0:
                angle_diff=90
            else:
                angle_diff=np.degrees(np.arctan(dy/dx))
            area_diff=abs(d1['w']*d1['h']-d2['w']*d2['h'])/(d1['w']*d1['h'])
            width_diff=abs(d1['w']-d2['w'])/d1['w']
            height_diff=abs(d1['h']-d2['h'])/d1['h']

            if distance<diagonal_length1*MAX_DIAG_MULTIPLYER\
            and angle_diff<MAX_ANGLE_DIFF and area_diff<MAX_AREA_DIFF\
            and width_diff<MAX_WIDTH_DIFF and height_diff<MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])
        if len(matched_contours_idx)<MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx=[]
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour=np.take(possible_contours,unmatched_contour_idx)

        recursive_contour_list=find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_contours_idx.append(idx)
        break

    return matched_result_idx

import pytesseract
#이미지 불러오기, gray로 바꾸기
img_ori=cv2.imread("test4_img.jpg")

height,width,channnel=img_ori.shape
plt.figure(figsize=(12,10))
plt.imshow(img_ori,cmap='gray')
plt.show()

gray=cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,10))
plt.imshow(gray,cmap='gray')
plt.show()

#이미지 필터링및 윤곽 잡기
img_blurred=cv2.GaussianBlur(gray,ksize=(9,1),sigmaX=0)

plt.figure(figsize=(20,20))
plt.imshow(img_blurred,cmap='gray')
plt.show()

img_thresh=cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

plt.figure(figsize=(20,20))
plt.imshow(img_thresh,cmap='gray')
plt.show()

#contours(윤곽을 묶는선)잡기
contours,_=cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result=np.zeros((height,width,channnel),dtype=np.uint8)
cv2.drawContours(temp_result,contours=contours,contourIdx=-1,color=(255,255,255))

plt.figure(figsize=(12,10))
plt.imshow(temp_result)
plt.show()

#contours를 사각형으로 묶기
contours_dict=[]

for contour in contours:
    x,y,w,h=cv2.boundingRect(contour)
    cv2.rectangle(temp_result,pt1=(x,y),pt2=(x+w,y+h),color=(255,255,255),thickness=2)

    contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx':x+(w/2),
        'cy':y+(h/2)
    })
plt.figure(figsize=(12,10))
plt.imshow(temp_result,cmap='gray')
plt.show()
#조건@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MIN_AREA=5000
MIN_WIDTH,MIN_HEIGHT=100,0
MIN_RATIO=13.0
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
possible_contours=[]

cnt=0
for d in contours_dict:
    area=d['w']*d['h']
    ratio=d['w']/d['h']
    #조건 실행
    if area>MIN_AREA and d['w']>MIN_WIDTH and d['h']>MIN_HEIGHT and MIN_RATIO<ratio:
        d['idx']=cnt
        cnt+=1
        possible_contours.append(d)

temp_result=np.zeros((height,width,channnel),dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(img_ori,pt1=(d['x'],d['y']),pt2=(d['x']+d['w'],d['y']+d['h']),color=(255,0,0),thickness=2)





plt.figure(figsize=(12,10))
plt.imshow(img_ori)
plt.show()


#조건2@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
MAX_DIAG_MULTIPLYER=10 #사각형 대각선길이 N배까지
MAX_ANGLE_DIFF=40.0   #사각형 두개의 각도 최대
MAX_AREA_DIFF=10000     #면적 크기차이
MAX_WIDTH_DIFF=10000   #width 크기차이
MAX_HEIGHT_DIFF=10000  #hight 크기차이
MIN_N_MATCHED=1       #그룹을 이루기위한 최소 갯수
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
result_idx=find_chars(possible_contours)

matched_result=[]
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours,idx_list))

temp_result=np.zeros((height,width,channnel),dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result,pt1=(d['x'],d['y']),pt2=(d['x']+d['w'],d['y']+d['h']),color=(255,255,255),thickness=2)

plt.figure(figsize=(12,10))
plt.imshow(temp_result,cmap='gray')
plt.show()


"""


















#이미지 위에서 보기

IMAGE_W = 1280

src = np.float32([[0, 700], [1280, 700], [0, 0], [1280, 0]])
dst = np.float32([[569, 700], [711, 700], [0, 0], [IMAGE_W, 0]])
M = cv2.getPerspectiveTransform(src, dst)    # The transformation matrix
Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


img = cv2.imread(r'C:\Users\pc\Downloads\itsea progect\ractangle\test4_img.jpg',cv2.IMREAD_COLOR)



# Read the test img
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()


img = img[450:(600), 0:IMAGE_W]      # Apply np slicing for ROI crop
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()

warped_img = cv2.warpPerspective(img, M, (IMAGE_W, 700)) # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))      # Show results
plt.show()


img_inv = cv2.warpPerspective(warped_img, Minv, (IMAGE_W, 700)) # Inverse transformation
plt.imshow(cv2.cvtColor(img_inv, cv2.COLOR_BGR2RGB))      # Show results
plt.show()





