import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
model = YOLO("yolov8n.pt")

#from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        print(self.P,"\n",self.K)
        print("==========================")
        self.cap = cv2.VideoCapture(os.path.join(data_dir, 'video.mp4'))
        self.orb = cv2.ORB_create(3000)
        self.images = []
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.cur_pose = np.array([[ 7.57556213e+02, -3.46027283e+02, -1.90984650e+02, -1.46613289e+04],
 [ 1.04178801e+01, -2.15563369e+00, -3.93643280e+02,  2.93432174e+03],
 [ 7.99507976e-01,  4.57323700e-01, -3.89412344e-01, -1.15028897e+01],[ 0.,0.,0.,1. ]])
        self.prec_pose = np.eye(4)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P



    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


    def get_matches(self, image_prec, image_cur):
        """
        This function detect and compute keypoints and descriptors from the previous image and current one using the class orb object

        Parameters
        ----------
        image_prec : the previous image
        image_cur : the current image

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
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

            for p in range(len(pts)):
                if  mask[int(pts[p].pt[1])][int(pts[p].pt[0])]<20:
                    pT.append(pts[p])
                    dT.append(desc[p])
            desc=np.array(dT)
            pts=tuple(pT)
            return pts,desc
        # Find the keypoints and descriptors with ORB
        kp1, des1 = getDescriptor(image_prec,self.orb)
        kp2, des2 = getDescriptor(image_cur, self.orb)
        # Find matches
        matches = self.flann.knnMatch(des1, des2, k=2)
        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(image_cur, kp1, image_prec,kp2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

def extract_rotation_angles(calibration_matrix):
    # Extrait la partie rotation de la matrice de calibration
    rotation_matrix = calibration_matrix[:3, :3]

    # Décomposition en valeurs singulières (SVD) pour obtenir la rotation
    U, _, Vt = np.linalg.svd(rotation_matrix)
    rotation = np.dot(U, Vt)

    # Calcul des angles d'Euler à partir de la matrice de rotation
    # Remarque : les angles sont en radians
    theta_x = np.arctan2(rotation[2, 1], rotation[2, 2])
    theta_y = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
    theta_z = np.arctan2(rotation[1, 0], rotation[0, 0])

    return [theta_x, theta_y, theta_z]

def visualize_map(points3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Afficher les points 3D de la carte
    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2], c='b', marker='o')




    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Map ')

    plt.show()


def main():
    data_dir = "videoGD"  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)



    gt_path = []
    estimated_path = []
    ANGLE = []
    ret,img_prec = vo.cap.read()
    img_prec=cv2.resize(img_prec,(640,360))
    vo.images.append(img_prec)

    #Lecture vidéo
    i=0
    while True:
        i+=1
        a=time.time()
        ret,img_cur = vo.cap.read()
        img_cur=cv2.resize(img_cur,(640,360))
        vo.images.append(img_cur)

        q1, q2 = vo.get_matches(img_prec,img_cur)
        transf = vo.get_pose(q1, q2)
        vo.cur_pose = np.matmul(vo.cur_pose, np.linalg.inv(transf))

        print("pose",vo.cur_pose)
        print("q1",q1)
        #print(extract_rotation_angles(cur_pose))
        ANGLE.append(extract_rotation_angles(vo.cur_pose))
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        # estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        #Points3D
        if i >=2:

            points4D_homogeneous = cv2.triangulatePoints(vo.prec_pose[:3], vo.cur_pose[:3], q1.T, q2.T)
            points3D_homogeneous = points4D_homogeneous / points4D_homogeneous[3]
            points3D = points3D_homogeneous[:3].T
            visualize_map(points3D)
        img_prec = img_cur #pas oublier d'actualiser l'image précédente.
        vo.prec_pose=vo.cur_pose
        print("temps",time.time()-a)
        #Pour quitter la boucle
        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        elif k == ord('q'):
            break


    print(ANGLE)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
