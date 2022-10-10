import cv2
import time 
import os
import HandTrackingModule as htm

wCam,Hcam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,Hcam)

folderPath = "D:\\python\\project001\\FingerImages"
myList = os.listdir(folderPath)
# print(myList)
overLayList=[]

for impath in myList:
    fra = cv2.imread(f"{folderPath}/{impath}")
    fra = cv2.resize(fra,(200,200))
    overLayList.append(fra)
pTime =0
detector = htm.handDetector(detectionCon=int(0.75))
Tipsids = [4,8,12,16,20]
while True:
    ret,frame = cap.read()
    frame = detector.findHands(frame,draw=True)
    lmlist,bbox = detector.findPosistion(frame,draw=True)
    # print(len(lmlist))
    if len(lmlist)!=0:
        fiingers=[]
        # print(Tipsids[0],Tipsids[0]-1)
        if lmlist[Tipsids[0]][1]>lmlist[Tipsids[0]-1][1]:
            fiingers.append(1)
        else:
            fiingers.append(0)
        for id in range(1,5):
            if lmlist[Tipsids[id]][2]<lmlist[Tipsids[id]-2][2]:
                fiingers.append(1)
            else:
                fiingers.append(0)
        # print(fiingers)
        totalFingers = fiingers.count(1)

        h,w,c = overLayList[totalFingers-1].shape
        # print(h,w)
        frame[0:h,0:w]=overLayList[totalFingers-1]

    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,f"FPS:{int(fps)}",(400,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,255),1)
    cv2.imshow("fra",frame)
    if cv2.waitKey(1)==ord('a'):
        break
cap.release()
cv2.destroyAllWindows()