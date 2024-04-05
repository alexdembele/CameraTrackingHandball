#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray

from ultralytics import YOLO
import cv2
import numpy as np

class YoloNode:
    def __init__(self):
        rospy.init_node('yolo_node', anonymous=True)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        self.bbox_pub = rospy.Publisher("/boundingbox/human", Float64MultiArray, queue_size=10)

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        frame = cv2.resize(frame, (640, 360))

        results = self.model(frame)  # predict on an image
        res_plotted = results[0].plot()
        cv2.imshow("result", res_plotted)
        cv2.waitKey(1)

        boxes = []
        for box in results[0].boxes.data:
            if box[5] < 0.5:
                boxes.append(list(box[:4]))
        box_msg = Float64MultiArray()
        box_msg.data = np.array(boxes).flatten().tolist()

        self.bbox_pub.publish(box_msg)
        print(np.array(boxes))

        

def main():
    yolo_node = YoloNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
