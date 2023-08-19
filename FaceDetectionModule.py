import cv2
import imutils
import mediapipe as mp
import time
import numpy as np


class FaceDetector():
    def __init__(self, mindetectConf=0.5):
        self.mindetectConf=mindetectConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.mindetectConf)

    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width*iw), int(bboxC.height*ih)
                bboxs.append([id,bbox,detection.score])
                img = self.fancyDraw(id,img,bbox)
                cv2.putText(img,f"Score:{round(detection.score[0]*100,2)}%",(bbox[0],bbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        return img, bboxs
    
    def fancyDraw(self,id,img,bbox):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        l=15
        t=2
        c=(0,255,0)
        cv2.rectangle(img,bbox,c,1)
        

        #top left
        cv2.line(img,(x,y),(x+l,y),c,t)
        cv2.line(img,(x,y),(x,y+l),c,t)
        # cv2.line(img,(x+l,y),(x,y+l),c,t)
        w=30
        h=17
        points=np.array([[x, y], [x+w, y], [x, y+h]])
        cv2.fillPoly(img,[points],(255,255,255))
        points=np.array([[x+w, y], [x, y+h],[x+w,y+h]])
        cv2.fillPoly(img,[points],(255,255,255))
        points=np.array([[x+w, y], [x+h+w, y],[x+w,y+h]])
        cv2.fillPoly(img,[points],(255,255,255))
        cv2.putText(img,f"{id}",(x+10,y+15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        
        #top right
        cv2.line(img,(x1,y),(x1-l,y),c,t)
        cv2.line(img,(x1,y),(x1,y+l),c,t)
        cv2.line(img,(x1-l,y),(x1,y+l),c,t)
        points=np.array([[x1,y],[x1-l,y],[x1,y+l]])
        cv2.fillPoly(img,[points],(255,255,255))
        
        #bottom left
        cv2.line(img,(x,y1),(x+l,y1),c,t)
        cv2.line(img,(x,y1),(x,y1-l),c,t)
        cv2.line(img,(x+l,y1),(x,y1-l),c,t)
        points=np.array([[x,y1],[x+l,y1],[x,y1-l]])
        cv2.fillPoly(img,[points],(255,255,255))
        
        #bottom right
        cv2.line(img,(x1,y1),(x1-l,y1),c,t)
        cv2.line(img,(x1,y1),(x1,y1-l),c,t)
        cv2.line(img,(x1-l,y1),(x1,y1-l),c,t)
        points=np.array([[x1-l,y1],[x1,y1-l],[x1,y1]])
        cv2.fillPoly(img,[points],(255,255,255))
        
        return img
        
    

    

def main():
    cap = cv2.VideoCapture("video/1.mp4")
    # cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = imutils.resize(img, width=960,height=540)
        # img=cv2.flip(img,1)
        img, bboxs = detector.findFaces(img)
        # print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
        cv2.imshow("FaceRec", img)
        cv2.waitKey(17)

if __name__ == "__main__":
    main()