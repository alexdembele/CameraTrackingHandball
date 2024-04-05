#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisherNode:
    def __init__(self):
        rospy.init_node('video_publisher_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        self.video_capture = cv2.VideoCapture(r"/home/dembele/Documents/1mnMontreau.mp4")  # Utilisez rospy.get_param pour récupérer le chemin de la vidéo
        
    def publish_video(self):
        rate = rospy.Rate(25)  # 25 images par seconde
        while not rospy.is_shutdown():
            
            ret, frame = self.video_capture.read()
            if ret:
                try:
                    frame = cv2.resize(frame,(640,360))
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.image_pub.publish(ros_image)
                except Exception as e:
                    rospy.logerr(e)
            rate.sleep()

if __name__ == '__main__':
    try:
        video_publisher_node = VideoPublisherNode()
        video_publisher_node.publish_video()
    except rospy.ROSInterruptException:
        pass
