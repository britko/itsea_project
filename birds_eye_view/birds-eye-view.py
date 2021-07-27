import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
# if cap.isOpen():
#     print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
	
while True:
    ret, fram = cap.read()
	
    if ret:
        img = cv2.cvtColor(fram, cv2.COLOR_BGR2BGRA)

        # rows, cols = img.shape[:2]

        # # 뒤집기 변환 행렬로 구현 ---①
        # st = time.time()
        # mflip = np.float32([ [-1, 0, cols-1],[0, -1, rows-1]]) # 변환 행렬 생성
        # fliped1 = cv2.warpAffine(img, mflip, (cols, rows))     # 변환 적용
        # print('matrix:', time.time()-st)

        # # remap 함수로 뒤집기 구현 ---②
        # st2 = time.time()
        # mapy, mapx = np.indices((rows, cols),dtype=np.float32) # 매핑 배열 초기화 생성
        # mapx = cols - mapx -1                                  # x축 좌표 뒤집기 연산
        # mapy = rows - mapy -1                                  # y축 좌표 뒤집기 연산
        # fliped2 = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)  # remap 적용
        # print('remap:', time.time()-st2)

        
        # [x,y] 좌표점을 4x2의 행렬로 작성
        # 좌표점은 좌상->좌하->우상->우하
        pts1 = np.float32([[504,1003],[243,1525],[1000,1000],[1280,1685]])

        # 좌표의 이동점
        pts2 = np.float32([[10,10],[10,1000],[1000,10],[1000,1000]])

        # pts1의 좌표에 표시. perspective 변환 후 이동 점 확인.
        # cv2.circle(img, (504,1003), 20, (255,0,0),-1)
        # cv2.circle(img, (243,1524), 20, (0,255,0),-1)
        # cv2.circle(img, (1000,1000), 20, (0,0,255),-1)
        # cv2.circle(img, (1280,1685), 20, (0,0,0),-1)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (1100,1100))
        
        # 결과 출력 ---③
        cv2.imshow('bird-eye-view', img)

        # q 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    else:
	    print('error')

cap.release()
cv2.destroyAllWindows()
