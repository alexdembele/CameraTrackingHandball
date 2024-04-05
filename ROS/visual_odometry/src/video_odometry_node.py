#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
import os
import numpy as np
import cv2
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometryNode():
    def __init__(self):

        rospy.init_node('visual_odometry_node')
        print(os.listdir())
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.pose_pub = rospy.Publisher("/camera/pose", Float64MultiArray, queue_size=10)
        self.K, self.P = self._load_calib("/home/dembele/catkin_ws/src/visual_odometry/config/calib.txt")
        self.orb = cv2.ORB_create(3000)
        self.images = []
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.cur_pose = np.array([[7.57556213e+02, -3.46027283e+02, -1.90984650e+02, -1.46613289e+04], [1.04178801e+01, -2.15563369e+00, -3.93643280e+02, 2.93432174e+03],[ 7.99507976e-01, 4.57323700e-01, -3.89412344e-01, -1.15028897e+01],[0.,0.,0.,1.]])
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
    
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.process_image(cv_image)
    
    def process_image(self, cv_image):
        if not hasattr(self, 'prev_image'):
            self.prev_image = cv_image
            return
        q1, q2 = self.get_matches(self.prev_image, cv_image)
        transf = self.get_pose(q1, q2)
        #print(transf)
        
        #anti erreur
        tampon  = np.matmul(self.cur_pose, np.linalg.inv(transf))
        print(tampon)
        if not np.isnan(tampon[0,0]):
            

            self.cur_pose = tampon
        #print(extract_rotation_angles(self.cur_pose))
        cv2.imshow("prev", self.prev_image)
        cv2.waitKey(1)
        self.prev_image = cv_image
        cv2.imshow("cur", cv_image)
        
        # Publish the current pose
        pose_msg = Float64MultiArray()
        pose_msg.data = self.cur_pose.flatten().tolist()
        self.pose_pub.publish(pose_msg)
       
        

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
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(image_prec, None)
        kp2, des2 = self.orb.detectAndCompute(image_cur, None)
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
        #cv2.imshow("image", img3)
        #cv2.waitKey(1)

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


def main():
    vo_node = VisualOdometryNode()
    rate = rospy.Rate(30)  # Définit la fréquence à 1 Hz (une itération par seconde)
    while not rospy.is_shutdown():
        # Votre code ici
        rate.sleep()





if __name__ == "__main__":
    main()
