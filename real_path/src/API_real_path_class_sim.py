#! /usr/bin/env python
import rospy
import math
import numpy as np
from eufs_msgs.msg import CarState
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion

class RealPathClass(object):
    def __init__(self):
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)

        self.real_path = Path()
        self.ground_path_pub = rospy.Publisher('/ground_path_pub', Path, queue_size=1)

        self.ground_pub = rospy.Publisher('/ground_pose_pub', PoseStamped, queue_size=1)
        self.marker_array_pub = rospy.Publisher('/ground_truth_mesh', Marker, queue_size=1)

        self.marker = Marker()
        self.pose = PoseStamped()

    def sub_callback(self, msg):
        self.real_path.header = msg.header
        self.real_path.header.frame_id = "map"
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        #yaw = self.pi_2_pi(np.arctan2(2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z), 1 - 2 * (msg.pose.pose.orientation.z ** 2)))
        #print('Ground truth yaw', yaw)

        self.real_path.poses.append(pose)
        self.marker_array(pose.pose)
        self.pose = pose

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def marker_array(self, pose):
        '''
        MarkerArray para mostrar en Rviz

        :param midpoint_x_list: Puntos en el eje x a mostrar
        :param midpoint_y_list: Puntos en el eje y a mostrar
        '''
        # Publish it as a marker in rviz
        self.marker.header.frame_id = "map"
        self.marker.ns = "est_pose_" + str(0)
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        self.marker.pose = pose
        self.marker.color.r, self.marker.color.g, self.marker.color.b = (255, 255, 255)
        self.marker.color.a = 0.5
        self.marker.scale.x, self.marker.scale.y, self.marker.scale.z = (1.0, 0.6, 0.4)
        # self.marker.mesh_resource = "../../eufs_sim-v1.0.0/eufs_description/meshes/chassis.dae"


if __name__ == '__main__':
    rospy.init_node('ground_path_node', anonymous=True)
    real_path_class = RealPathClass()
    #rospy.spin()
    rate = rospy.Rate(50)  # Hz

    while not rospy.is_shutdown():
        real_path_class.ground_path_pub.publish(real_path_class.real_path)
        real_path_class.ground_pub.publish(real_path_class.pose)
        real_path_class.marker_array_pub.publish(real_path_class.marker)
        rate.sleep()