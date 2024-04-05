#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import numpy as np

class PoseConverterNode:
    def __init__(self):
        rospy.init_node('pose_converter_node', anonymous=True)

        # Subscriber for camera pose
        self.camera_pose_sub = rospy.Subscriber("/camera/pose", Float64MultiArray, self.camera_pose_callback)

        # Subscriber for bounding box human
        self.bounding_box_sub = rospy.Subscriber("/boundingbox/human", Float64MultiArray, self.bounding_box_callback)

        # Publisher for point cloud
        self.point_cloud_pub = rospy.Publisher("/humancloud", PointCloud, queue_size=10)

        self.pose_inverse = np.eye(4)

    def camera_pose_callback(self, data):
        # Convert Float64MultiArray to numpy array
        pose_data = np.array(data.data)
        
        # Reshape the numpy array to 4x4 matrix
        pose_matrix = pose_data.reshape(4, 4)



        self.pose_inverse = pose_matrix[:3,:]

    def bounding_box_callback(self, data):
        # Convert Float64MultiArray to numpy array
        bounding_box_data = np.array(data.data)

        #nombre de bounding box
        nb_box = len(bounding_box_data)//4
        bbox = np.reshape(bounding_box_data,(nb_box,4))
        point_cloud_msg = PointCloud()

        for i in range(len(bbox)):
            point = Point32()
            #projection inverse pour avoir pied humain sur le terrain
            X = np.array([bbox[i,2],bbox[i,3],0,1])
            Y = self.pose_inverse @ X


            point.x = Y[0]
            point.y = Y[1]
            point.z = 0.0  # Assuming 2D points, so z = 0
            point_cloud_msg.points.append(point)
            
            


        

        self.point_cloud_pub.publish(point_cloud_msg)

    def inverse_camera_matrix(P):
        # Extract rotation matrix R and translation vector t from P
        K, RT = np.split(P, [3], axis=1)
        R, t = np.split(RT, [3], axis=1)

        # Compute inverse of rotation matrix (transpose)
        R_inv = np.transpose(R)

        # Compute inverse translation vector
        t_inv = -np.dot(R_inv, t)

        # Construct the inverse camera matrix P_inv
        P_inv = np.concatenate((np.dot(K, np.concatenate((R_inv, t_inv), axis=1))), axis=1)
        
        return P_inv
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        pose_converter_node = PoseConverterNode()
        pose_converter_node.run()
    except rospy.ROSInterruptException:
        pass
