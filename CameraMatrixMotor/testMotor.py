import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from CameraMatrixMotor import *
from scipy.optimize import minimize


#Import image
JesseOwens=cv2.imread(r"Image\terrainCouleur2.png")
JesseOwens=cv2.resize(JesseOwens,(640,360))
homo=cv2.imread(r"Image\homographie.png")
homo=cv2.resize(homo,(640,360))
newHomo=np.zeros((1000,1900,3),dtype=np.uint8)
newHomo[200:560,300:940]=homo
newJesse=np.zeros((1000,1900,3),dtype=np.uint8)
newJesse[200:560,300:940]=JesseOwens

b,g,r=cv2.split(homo)
imgBleu=np.copy(homo)
imgBleu[r>145]=[0,0,0]
imgBleu[b<100]=[0,0,0]


imgBis=[]
for i in range(360):
    for j in range(640):
        imgBis.append([j,i,1])
homoresh=np.reshape(homo,(640*360,3))
homobleu=np.reshape(imgBleu,(640*360,3))
imgBis=np.transpose(np.array(imgBis,dtype=np.float64))



#set matrixEngine


Camera = CameraEngine()

Camera.setCalibration(451.08695555*1.5,449.41858171*0.8,347.86504384*1.5,178.48101762*0.9)


Camera.setCenter([-20,3,-8])
Camera.setTilt(-np.pi/2)

#definition image
coords=[]
for i in range(360):
    for j in range(640):


        coords.append([j*40/640,(360-i)*20/360,0,1])

coords=np.transpose(np.array(coords,dtype=np.float64))
largeur=1000
longueur=1900
#Mouvememnt clavier
pan=1.117550186681535
tilt=-3.616576076205223
roll=-1.3320352851220814
Camera.setPan(pan)
Camera.setRoll(roll)
Camera.setTilt(tilt)
print("P",Camera.P)










### controle clavier
while True:
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k==ord("q"):
        pan+=np.pi/1000
    elif k==ord("d"):
        pan-=np.pi/1000
    elif k==ord("z"):
        tilt+=np.pi/1000
    elif k==ord("s"):
        tilt-=np.pi/1000
    elif k==ord("l"):
        roll+=np.pi/1000
    elif k==ord("m"):
        roll-=np.pi/1000


    Camera.setPan(pan)
    Camera.setRoll(roll)
    Camera.setTilt(tilt)
    I=np.copy(coords)
    Y=Camera.P@I
    Y=Y/Y[2]

    ig=np.zeros((largeur,longueur,3),dtype=np.uint8)
    Y=Y.astype(np.uint32)


    min0=np.zeros_like(Y[2,:])
    max640=min0+1899
    max360=min0+999

    Y[0,:]=np.maximum(Y[0,:]+300,min0)
    Y[1,:]=np.maximum(Y[1,:]+200,min0)
    y_src=np.minimum(Y[0,:],max640)
    x_src=np.minimum(Y[1,:],max360)



    A,B=np.meshgrid(range(640),range(360))

    ig[x_src,y_src]=JesseOwens[B.flatten(),A.flatten()]
    img=cv2.addWeighted(ig, 0.7, newHomo, 0.5, 0)
    cv2.imshow("gpt",img)

    cv2.waitKey(1)


