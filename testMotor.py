import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import time

from CameraMatrixMotor import *
from scipy.optimize import minimize


#Import image
JesseOwens=cv2.imread(r"..\Image\terrainCouleur2.png")
JesseOwens=cv2.resize(JesseOwens,(640,360))
homo=cv2.imread(r"..\Image\homographie.png")
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

def distance(i,j,numero):
    '''
    Calcul la distance entre un pixel i,j de l'image (apres homographie ) et le terrain
    Le parametre numero nous indique quelle distance on calcule
    '''
    imin=212
    imax=536
    jmin=320
    jmax=920
    if i<=imin and j<=jmin: #On est en haut à gauche hors du terrain
        d=math.dist((i,j),(imin,jmin)) #On calcule la distance avec le coin haut gauche du terrain
    elif i>=imax and j <=jmin:#On est en bas à gauche hors du terrain
        d=math.dist((i,j),(imax,jmin))#On calcule la distance avec le coin bas gauche du terrain
    elif i>imin and i<imax and j<=jmin:#On est au milieu à gauche hors du terrain
        d=math.dist((i,j),(i,jmin))#On calcule la distance avec la gauche du terrain
    elif i<=imin and j>=jmax: #On est en haut à droite hors du terrain
        d=math.dist((i,j),(imin,jmax))#On calcule la distance avec le coin haut droite du terrain
    elif i>=imax and j >=jmax: #On est en bas à droite hors du terrain
        d=math.dist((i,j),(imax,jmax))#On calcule la distance avec le coin bas droite du terrain
    elif i>imin and i<imax and j>=jmax:#On est au milieu à droite hors du terrain
        d=math.dist((i,j),(i,jmax))#On calcule la distance avec la droite du terrain
    elif i<=imin and j>jmin and j <jmax:#On est en haut au milieu hors du terrain
        d=math.dist((i,j),(imin,j))#On calcule la distance avec le  haut  du terrain
    elif i>=imax and j>jmin and j <jmax:#On est en bas au milieu  hors du terrain
        d=math.dist((i,j),(imax,j))#On calcule la distance avec le  bas  du terrain
    elif i>imin and i <imax and j>jmin and j<jmax:#On est sur le terrain
        if numero==1:
            return 1
        else:

            return 0
    else:

        d=1

    if numero==1:
        return math.exp(-d)
    else:
        return math.exp(-(1/(d+1)))


def distColor(i,j,colorImg,newJesse):
    if i<212 or i>536 or j<320 or j >920: #On regarde si on est sur le terrain,
        return math.dist(colorImg,[0,0,0]) #Si non, on renvoie l'écart de couleur à un pixel noir
    else:

        return math.dist(colorImg,newJesse[i,j]) #Si oui, on renvoie l'écart de couleur au pixel correspondant du terrain

lamb=1000
def lossVianney():
    '''
    somme distance(pixel bleu,centre du terrain) + somme(comparaison couleur)(1-distance entre pixels)
    Les sommes se font sur les pixels de la source
    On warp l'image de la video
    '''

    #Nouvelle dimensions de l'image

    #tableau pour calcul de la loss
    coords1=np.zeros((homo.shape[0] * homo.shape[1], 8),dtype=np.uint32)
    coords1[:,0]=x_src
    coords1[:,1]=y_src
    coords1[:,2]=homoresh[:,0]
    coords1[:,3]=homoresh[:,1]
    coords1[:,4]=homoresh[:,2]
    coords1[:,5]=homobleu[:,0]
    coords1[:,6]=homobleu[:,1]
    coords1[:,7]=homobleu[:,2]
    mask=np.all(coords1[:,5:8]!=[0,0,0],axis=1)
    S1,S2=0,0
    t1 = cv2.getTickCount()

    #Correspondance des couleurs pondéré par la distance au terrain
    S1=np.sum(np.array([distColor(int(pixel[0]),int(pixel[1]),pixel[2:5],newJesse)*distance(int(pixel[0]), int(pixel[1]), 1) for pixel in coords1]))

    #Distance au terrain des pixels bleu
    S2=np.sum(np.array([distance(int(pixel[0]), int(pixel[1]), 2) for pixel in coords1[mask]]))
    print("S1 :",S1,"S2 :",S2)


    return S1+lamb*S2,S1,S2

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


### minimisation scipy





# Vecteur initial de valeurs
vecteur_initial = [pan,tilt,roll]
def objective(vecteur):

    pan,tilt,zoom=vecteur
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

    coords1=np.zeros((homo.shape[0] * homo.shape[1], 8),dtype=np.uint32)
    coords1[:,0]=x_src
    coords1[:,1]=y_src
    coords1[:,2]=homoresh[:,0]
    coords1[:,3]=homoresh[:,1]
    coords1[:,4]=homoresh[:,2]
    coords1[:,5]=homobleu[:,0]
    coords1[:,6]=homobleu[:,1]
    coords1[:,7]=homobleu[:,2]
    mask=np.all(coords1[:,5:8]!=[0,0,0],axis=1)
    S1,S2=0,0
    t1 = cv2.getTickCount()

    #Correspondance des couleurs pondéré par la distance au terrain
    S1=np.sum(np.array([distColor(int(pixel[0]),int(pixel[1]),pixel[2:5],newJesse)*distance(int(pixel[0]), int(pixel[1]), 1) for pixel in coords1]))

    #Distance au terrain des pixels bleu
    S2=np.sum(np.array([distance(int(pixel[0]), int(pixel[1]), 2) for pixel in coords1[mask]]))



    return S1+lamb*S2



# Minimisation de la fonction
debut = time.time()
resultat = minimize(objective, vecteur_initial)
print("temps = ",time.time()-debut)


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
    loss,S1,S2=lossVianney()
    cv2.imshow("gpt",img)

    cv2.waitKey(1)


