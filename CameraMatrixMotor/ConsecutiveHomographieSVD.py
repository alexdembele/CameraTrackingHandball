#Consecutive homography computation in a video
#Using SVD to determinate consecutive homography

import numpy as np
import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
#Terrain de référence
JesseOwens=cv2.imread(r'../Image/JesseOwens.png')
JesseOwens=cv2.cvtColor(JesseOwens,cv2.COLOR_BGR2GRAY)

kp = cv2.KAZE_create(upright = False,#Par défaut : false
            threshold = 0.001,#Par défaut : 0.001
            nOctaves = 4,#Par défaut : 4
        nOctaveLayers = 4,#Par défaut : 4
        diffusivity = 2)#Par défaut : 2



color = (255, 0, 0)
thickness = -1

def getDescriptor(img,kp):
    img=cv2.resize(img,(640,360))
    results=model(img)
    mask=np.zeros((360,640))
    for box in results[0].boxes.data:
        if box[5]<0.5:
            cv2.rectangle(mask,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),255,-1)
    gray =  cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pts, desc = kp.detectAndCompute(gray,None)
    pT=[]
    dT=[]
    cv2.imshow("mask",mask)
    for p in range(len(pts)):
        if  mask[int(pts[p].pt[1])][int(pts[p].pt[0])]<20:
            pT.append(pts[p])
            dT.append(desc[p])
    desc=np.array(dT)
    pts=tuple(pT)
    return pts,desc
cap=cv2.VideoCapture(r"..\vidéo\1mnMontreau.mp4")

#Première homographie (calibrage)
H=np.array([[-1.14034673e+00, -7.24754754e+00,  1.90271130e+01],
            [-1.27700509e+00, -9.65913629e+00,  1.43703344e+03],
            [-2.33345791e-04, -2.39702046e-02,  1.00000000e+00]])



ret, img1 = cap.read()
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray1=cv2.resize(gray1,(640,360))
pts1,desc1=getDescriptor(img1,kp)
grayWarp=cv2.warpPerspective(gray1,H, (JesseOwens.shape[1], JesseOwens.shape[0]))
result= cv2.addWeighted(grayWarp, 0.7, JesseOwens, 0.5, 0)
#cv2.imshow('1ere',result)

while True:
    ret,img2=cap.read()
    gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray2=cv2.resize(gray2,(640,360))
    pts2,desc2=getDescriptor(img2,kp)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    if pts1!=() and pts2!=():
        matches = bf.knnMatch(desc1,desc2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.3*n.distance:
                good.append([m])

        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        flags = 0)
        img3 = cv2.drawMatchesKnn(gray1,pts1,gray2,pts2,good,None,**draw_params)

        A=np.zeros((2*len(good),9))

        #SVD
        for i in range(len(good)-1):

            Xs=pts1[good[i][0].queryIdx].pt
            Xd=pts2[good[i][0].trainIdx].pt
            A[2*i]=[Xs[0],Xs[1],1,0,0,0,-Xs[0]*Xd[0],-Xs[1]*Xd[0],-Xd[0]]
            A[2*i+1]=[0,0,0,Xs[0],Xs[1],1,-Xs[0]*Xd[1],-Xs[1]*Xd[1],-Xd[1]]

        U,S,V=np.linalg.svd(-A)
        V1=V[-1]
        H_SVD=np.reshape(V1,(3,3))
        H=H@H_SVD
        homo = cv2.warpPerspective(gray2,H, (640, 360),flags=cv2.INTER_NEAREST)

        imaginaire = cv2.addWeighted(homo, 0.7, JesseOwens, 0.5, 0)
        cv2.imshow("homographie continue",imaginaire)
        Nb_ok = len(good)


        cv2.imshow('appariement',img3)

    gray1=np.copy(gray2)
    pts1=pts2
    desc1=desc2
    keyboard = cv2.waitKey(30)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()