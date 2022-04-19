#! /usr/bin/env python

import rospy
import numpy as np
import copy

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point

class Local2global():
    def __init__(self):

        self.global_path = Path()
        self.path = Path()
        self.global_path_planing = Path()
        self.path_planing = Path()
        self.yaw_init = -0.5

        # Subscribers

        self.path_sub = rospy.Subscriber('/path_pub', Path, self.path_callback)
        self.path_planning_sub = rospy.Subscriber('/path_planning_pub', Path, self.path_planning_callback)

        # Publishers

        self.global_path_pub = rospy.Publisher('/path_pub_global', Path, queue_size=1)
        self.global_path_planning_pub = rospy.Publisher('/path_planning_pub_global', Path, queue_size=1)

    def path_callback(self,msg):
        self.path = msg
        self.path_local2global()

    def path_planning_callback(self,msg):
        self.path_planing=msg
        self.path_planning_local2global()

    def path_local2global(self):
        '''
        Transforma las coordenadas locales de path en globales para visualizar el simulador
        '''
        pos = PoseStamped()
        point = Point()
        self.global_path.header = copy.deepcopy(self.path.header)
        if len(self.path.poses)>0:
            pos = copy.deepcopy(self.path.poses[-1])
            point = copy.deepcopy(self.path.poses[-1].pose.position)
            
            pos.pose.position.x = point.x*np.math.cos(self.yaw_init)-point.y*np.math.sin(self.yaw_init)
            pos.pose.position.y = point.x*np.math.sin(self.yaw_init)+point.y*np.math.cos(self.yaw_init)
            pos.pose.position.z = point.z
            self.global_path.poses.append(copy.copy(pos))

    def path_planning_local2global(self):
        '''
        Transforma las coordenadas locales de path planning en globales para visualizar el simulador
        '''
        pos = PoseStamped()
        point = Point()
        self.global_path_planing.header = copy.deepcopy(self.path_planing.header)
        if len(self.path_planing.poses)>0:
            pos = copy.deepcopy(self.path_planing.poses[-1])
            point = copy.deepcopy(self.path_planing.poses[-1].pose.position)
            
            pos.pose.position.x = point.x*np.math.cos(self.yaw_init)-point.y*np.math.sin(self.yaw_init)
            pos.pose.position.y = point.x*np.math.sin(self.yaw_init)+point.y*np.math.cos(self.yaw_init)
            pos.pose.position.z = point.z
            self.global_path_planing.poses.append(copy.copy(pos))


        # if len(self.path.poses)>len(self.global_path.poses) and len(self.path.poses)>0:
        #     self.length = len(self.path.poses)
        #     self.n = len(self.path.poses)-len(self.global_path.poses)
        #     for i in range(self.length-self.n, self.length):
        #         pos = PoseStamped()
        #         point = Point()
        #         self.global_path.header = copy.deepcopy(self.path.header)
        #         pos = copy.deepcopy(self.path.poses[i])
        #         point = copy.deepcopy(self.path.poses[i].pose.position)
                
        #         pos.pose.position.x = point.x*np.math.cos(self.yaw_init)-point.y*np.math.sin(self.yaw_init)
        #         pos.pose.position.y = point.x*np.math.sin(self.yaw_init)+point.y*np.math.cos(self.yaw_init)
        #         pos.pose.position.z = point.z
        #         self.global_path.poses.append(copy.copy(pos))

if __name__ == '__main__':
    rospy.init_node('local2global_node', anonymous=True)
    local2global_class = Local2global()
    rate = rospy.Rate(50)  # Frecuencia de los publishers (Hz)

    while not rospy.is_shutdown():
        
        local2global_class.global_path_pub.publish(local2global_class.global_path)
        local2global_class.global_path_planning_pub.publish(local2global_class.global_path_planing)

        rate.sleep()
