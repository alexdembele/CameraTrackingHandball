import numpy as np
import cv2 as cv

class CameraEngine:

    def __init__(self):
        self.P = np.zeros((3, 4)) # projection matrix
        self.RotationRoll=np.eye(3)
        self.RotationTilt=np.eye(3)
        self.RotationPan=np.eye(3)
        self.K=np.zeros((3,3)) #Calibration matrix
        self.Center=np.zeros(3)

    def computeMatrix(self):
        self.P = np.hstack((np.eye(3),np.reshape(self.Center,(3,1))))
        self.P = self.K @ self.RotationRoll @ self.RotationTilt @ self.RotationPan @  self.P


    def setRoll(self,roll):
        R=np.array([[np.cos(roll),-np.sin(roll),0],[np.sin(roll),np.cos(roll),0],[0,0,1]],dtype=np.float32)
        self.computeMatrix()


    def setPan(self,pan):
        self.RotationPan=np.array([[np.cos(pan),0,np.sin(pan)],[0,1,0],[-np.sin(pan),0,np.cos(pan)]],dtype=np.float32)
        self.computeMatrix()


    def setTilt(self,tilt):
        self.RotationTilt=np.array([[1,0,0],[0,np.cos(tilt),-np.sin(tilt)],[0,np.sin(tilt),np.cos(tilt)]],dtype=np.float32)
        self.computeMatrix()

    def setCenter(self,coords):
        self.Center[0]=coords[0]
        self.Center[1]=coords[1]
        self.Center[2]=coords[2]
        self.computeMatrix()

    def setCalibration(self,f1,f2,a1,a2):
        self.K = np.array([[f1,0,a1],[0,f2,a2],[0,0,1]],dtype=np.float32)
        self.computeMatrix()

    def keyboardControl(self):



