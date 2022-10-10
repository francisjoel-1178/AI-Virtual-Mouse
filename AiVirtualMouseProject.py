import cv2
import HandTrackingModule as htm
import numpy as np
import time
import autopy
#######
wCam,hCam = 640,480
frameR=100
smoothening = 7
########
pTime = 0
pLocx,pLocy = 0,0
cLocx,cLocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(maxHands=1)
wScr,hScr = autopy.screen.size()

while True:
    # 1.Find hand landmarks
    ret,frame = cap.read()
    frame = detector.findHands(frame=frame)
    lmlist,bbox= detector.findPosistion(frame,draw=True)
    # 2.Get the tip of index and middle finger
    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        # print(x1,y1,x2,y2) 
        # 3.check fingers are up
        fingers = detector.fingersUP()
        # print(fingers) 
        cv2.rectangle(frame,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

        # 4.Only index Finger:moving mode
        if fingers[1]==1 and fingers[2]==0:
            # 5.convert coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            # 6.Smoothening values
            cLocx = pLocx+(x3-pLocx)/smoothening
            cLocy = pLocy+(x3-pLocy)/smoothening
            # 7.moving mouse
            autopy.mouse.move(wScr-cLocx,cLocy)
            cv2.circle(frame,(x1,y1),12,(255,0,255),cv2.FILLED)
            pLocx,pLocy = cLocx,cLocy
        # 8.Both middle and index fingers up: clicking mode
        if fingers[1]==1 and fingers[2]==1:
            length,frame,lineInfo= detector.findDistance(8,12,frame)
            # print(length)
            # 9.mouse clicking if length is less than 40
            if length<40:
                cv2.circle(frame,(lineInfo[4],lineInfo[5]),12,(0,255,0),cv2.FILLED)
                autopy.mouse.click()
    # print(bbox)
    # 11. fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    # 12. display
    cv2.imshow("frmae",frame)
    if cv2.waitKey(1)==ord('a'):
        break
cap.release()
cv2.destroyAllWindows()
