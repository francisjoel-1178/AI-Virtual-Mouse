import cv2
import mediapipe as mp
import time
import math

class handDetector:
    def __init__(self,mode=False,maxHands=2,detectionCon=int(0.5),trackCon=int(0.5)):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands = mp.solutions.hands
        self.hands  = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self,frame,draw=True):

        rgbFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgbFrame)
        

        if self.results.multi_hand_landmarks:
            for handlms in  self.results.multi_hand_landmarks:
                if draw:      
                    self.mpDraw.draw_landmarks(frame,handlms,self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosistion(self,frame,handNo=0,draw=True):
        xlist=[]
        ylist=[]
        bbox=[]
        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id,lms in enumerate(myHand.landmark):       
                h,w,c = frame.shape
                cx,cy = int(lms.x*w),int(lms.y*h)
                # print(id,cx,cy)
                xlist.append(cx)
                ylist.append(cy)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,25,255),cv2.FILLED)
            xmin,xmax = min(xlist),max(xlist)
            ymin,ymax = min(ylist),max(ylist)
            if draw:
                cv2.rectangle(frame,(xmin-20,ymin-20),(xmax+20,ymax+20),(255,255,255),2)
            bbox = xmin, ymin, xmax, ymax
         
        return self.lmlist,bbox

    def fingersUP(self):
        fingers=[]
        # print(len(self.lmlist))
        if self.lmlist[self.tipIds[0]][1]> self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

                            
        # 4 FIngers
        for Id in range(1,5):
            if self.lmlist[self.tipIds[Id]][2]<self.lmlist[self.tipIds[Id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self,p1,p2,frame,draw=True,r=15,t=3):

        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        if draw:
            cv2.line(frame,(x1,y1),(x2,y2),(225,0,255),t)
            cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
            cv2.circle(frame,(x2,y2),r,(255,0,255),cv2.FILLED)
            # cv2.line(frame,(x1,y1),(x2,y2),(225,0,255),3)
            cv2.circle(frame,(cx,cy),r,(255,0,255),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)

        return length, frame, [x1,x2,y1,y2,cx,cy]

    


             
     
def main():
    
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret,frame = cap.read()
        frame = detector.findHands(frame=frame)
        lmlist,bbox = detector.findPosistion(frame=frame)
        if len(lmlist)!=0:
            fingers = detector.fingersUP()
            # print(fingers)
        
        

        cTime=time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime

        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
        
        
        cv2.imshow("framen",frame)
        

        cv2.waitKey(1)

if __name__=="__main__":
    main()
# main()