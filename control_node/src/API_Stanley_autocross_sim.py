#! /usr/bin/env python

from std_msgs.msg import Bool
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path
from eufs_msgs.msg import CarState
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np
import rospy
import Algoritmo_Stanley


class State(object):
    def __init__(self):
        self.stanley_class = Algoritmo_Stanley.Stanley()

        self.gps_vel = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.callbackgps)
        self.path_planning_sub = rospy.Subscriber('/path_planning_pub', Path, self.callbackpath)
        # self.real_path = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.sub_callback2)
        # self.state_estimation_sub = rospy.Subscriber('/slam_pose_pub', PoseStamped, self.sub_callback2)
        self.control = rospy.Publisher('/cmd_vel_out', AckermannDriveStamped, queue_size=1)
        self.start = rospy.Publisher('/ros_can/mission_flag', Bool, queue_size=1)
        self.finish=rospy.Publisher('/Finish',Int16,queue_size=1)
        self.cot_vuelta=rospy.Publisher('/cont_vuelta',Int16,queue_size=1)

        self.ack_msg = AckermannDriveStamped()
        self.ack_msg.header.frame_id = "map"
        self.ack_msg.drive.speed = 0  # [m/s]
        self.ack_msg.drive.steering_angle = 0.0  # [rad]
        self.ack_msg.drive.steering_angle_velocity = 0  # [rad/s]
        self.ack_msg.drive.acceleration = 0.0  # [m/s^2]
        self.ack_msg.drive.jerk = 0  # [m/s^3]

    def callbackgps(self, data):
        v = np.sqrt(data.vector.x ** 2 + data.vector.y ** 2)  # [m/s]
        if v == 0:
            self.start.publish(True)      # to go from state OFF to state DRIVING
        self.stanley_class.update_velocity(v)

    def sub_callback(self, msg):
        self.stanley_class.update_car_position(msg.pose.pose)

    def sub_callback2(self, msg):
        self.stanley_class.update_car_position(msg.pose)

    def callbackpath(self, data):
        self.ack_msg.header.stamp = rospy.Time.now()
        self.stanley_class.principal_loop(data)


if __name__ == '__main__':
    rospy.init_node('stanley', anonymous=True)
    state = State()
    rate = rospy.Rate(200)  # Hz

    while not rospy.is_shutdown():
        state.ack_msg.drive.acceleration = state.stanley_class.acceleration
        state.ack_msg.drive.steering_angle = state.stanley_class.steering_angle
        state.control.publish(state.ack_msg)
        state.cot_vuelta.publish(state.stanley_class.contador_de_vuelta)
        if state.stanley_class.finish_flag==1:
            state.finish.publish(state.stanley_class.finish_flag)
        rate.sleep()
