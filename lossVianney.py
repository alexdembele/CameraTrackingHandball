import numpy as np

def lossVianney(H,):
    '''
    somme distance(pixel bleu,centre du terrain) + somme(comparaison couleur)(1-distance entre pixels)
    Les sommes se font sur les pixels de la source
    On warp l'image de la video
    '''

    #Nouvelle dimensions de l'image
    largeur=1000
    longueur=1900
    ig=np.zeros((largeur,longueur,3),dtype=np.uint8)

    #Application de l'homographie , calcul des nouvelles coordonnées des pixels
    Y=np.copy(imgBis)
    Y=np.dot(H,Y)
    Y=Y/Y[2,:]
    Y=Y.astype(np.uint32) #conversion des coordonnées flottantes en entiers pour pouvoir être des indices

    #Verification de l'appartenance au cadre de la nouvelle image
    min0=np.zeros_like(Y[2,:])
    max640=min0+1899
    max360=min0+999
    Y[0,:]=np.maximum(Y[0,:]+300,min0) #Si on est hors cadre, on envoie le pixel sur le bord de l'image
    Y[1,:]=np.maximum(Y[1,:]+200,min0)
    y_src=np.minimum(Y[0,:],max640)
    x_src=np.minimum(Y[1,:],max360)
    image=np.reshape(homoresh,(360,640,3))

    Y,X=np.meshgrid(range(640),range(360))
    ig[x_src,y_src]=image[X.flatten(),Y.flatten()]# On remplit l'image cible en envoyant les pixels sur leurs nouvelles coordonnées
    img = cv2.addWeighted(ig, 0.7, newJesse, 0.5, 0)
    cv2.imshow("gpt",img)
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