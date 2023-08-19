import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self,mode=False,maxHands=2,detectConf=1,trackConf=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectConf=detectConf
        self.trackConf=trackConf
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
            
        return img
  
    def findTipOfFinger(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    lmList.append([id,cx,cy])
                    if draw:
                        if id==0 or id==4 or id==8 or id==12 or id==16 or id==20:
                            cv2.circle(img,(cx,cy),30,(100,30,150),3)
        return lmList
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        list = detector.findTipOfFinger(img)
        
        if len(list)!=0:
            print(list[0])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        #frame,text,position,fontstyle,scale,color(bgr),thickness
        cv2.putText(img,str(int(fps)),(50,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),4)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()