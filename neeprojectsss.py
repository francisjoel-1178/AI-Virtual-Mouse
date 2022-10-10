import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands  = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    ret,frame = cap.read()
    rgbFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgbFrame)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in  results.multi_hand_landmarks:
            for id,lms in enumerate(handlms.landmark):
                # print(id,lms) 
                h,w,c = frame.shape
                cx,cy = int(lms.x*w),int(lms.y*h)
                # print(id,cx,cy)
                # if id==0:
                cv2.circle(frame,(cx,cy),20,(255,25,255),cv2.FILLED)
            mpDraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS)
    
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    
    
    cv2.imshow("framen",frame)
    

    if cv2.waitKey(1)==ord("a"):
        break;
cap.release()
cv2.destroyAllWindows()