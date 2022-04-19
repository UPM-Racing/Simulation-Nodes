#! /usr/bin/env python
import rospy

from eufs_msgs.msg import ConeArrayWithCovariance, CarState
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import pathplanning_EUFS as pathplanning

class Path_planning_class(object):
    def __init__(self):
        # Inicializacion de variables
        self.path_planning = pathplanning.Path_planning()

        ''' Topicos de ROS '''
        # Subscriber de las observaciones de conos
        self.cones_sub = rospy.Subscriber('/ground_truth/cones', ConeArrayWithCovariance, self.cones_callback)

        # Subscriber para la actualizacion de la posicion del coche (Solo debe estar activa 1)
        # self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.ground_truth_callback)
        self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.state_callback)
        # self.slam_pose_sub = rospy.Subscriber('/slam_pose_pub', PoseStamped, self.state_callback)

        # Publishers de path planning
        self.path_pub = rospy.Publisher('/path_planning_pub', Path, queue_size=1)
        self.marker_array_pub = rospy.Publisher('/path_planning_marker_array_pub', MarkerArray, queue_size=1)

    def ground_truth_callback(self, msg):
        self.path_planning.update_car_position(msg.pose.pose)

    def state_callback(self, msg):
        self.path_planning.update_car_position(msg.pose)

    def cones_callback(self, msg):
        cones_yellow = msg.yellow_cones
        cones_blue = msg.blue_cones
        cones_orange = msg.big_orange_cones
        if not self.path_planning.finished_lap:
            self.path_planning.bucle_principal(cones_yellow, cones_blue, cones_orange)


if __name__ == '__main__':
    rospy.init_node('path_planning_node', anonymous=True)
    path_class = Path_planning_class()
    rate = rospy.Rate(10)  # Frecuencia de los publishers (Hz)

    while not rospy.is_shutdown():
        path_class.marker_array_pub.publish(path_class.path_planning.marker_ests)
        path_class.path_pub.publish(path_class.path_planning.path)
        rate.sleep()
