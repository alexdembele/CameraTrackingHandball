#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

print("Acquisition Camera Python ...")

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/video/raw', Image, self.callback)

    def callback(self, data):       
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            
            cv2.imshow("USB Camera", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            
            print(e)

if __name__ == '__main__':
    
    try:
        
        image_subscriber = ImageSubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
