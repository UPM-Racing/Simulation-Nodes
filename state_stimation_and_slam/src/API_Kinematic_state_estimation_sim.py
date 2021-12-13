#! /usr/bin/env python
import rospy
import numpy as np
import math

from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix

from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3Stamped
from eufs_msgs.msg import WheelSpeedsStamped
from visualization_msgs.msg import MarkerArray, Marker
from GPS import GPS

import Kinematic_state_estimation as state_estimation

class EKF_Class(object):
    def __init__(self):
        # Inicializacion de variables
        self.car = state_estimation.Car()
        self.gps = GPS()
        self.first = 2

        ''' Topicos de ROS '''
        # Subscriber de la entrada de control para el modelo cinematico
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)

        # Subscribers para corregir el estado del coche
        self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)
        # self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)
        self.odom_sub = rospy.Subscriber('/ros_can/wheel_speeds', WheelSpeedsStamped, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Publishers de state estimation
        self.path_pub = rospy.Publisher('/path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/pose_pub', PoseStamped, queue_size=1)
        self.marker_array_pub = rospy.Publisher('/covariance_ellipse_pub', MarkerArray, queue_size=1)
        self.control_for_slam_pub = rospy.Publisher('/control_for_slam', WheelSpeedsStamped, queue_size=1)

    def control_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        u = np.zeros(2)
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration
        u[0] = acceleration
        u[1] = angle
        self.car.control_msg.steering = angle

        if self.first != 0:
            self.car.InicializacionCoche(time - 0.02)
            self.first -= 1
        self.car.Kinematic_prediction(time, u)

    def gps_callback(self, msg):
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude
        x, y = self.gps.gps_to_local(latitude, longitude, altitude)
        gps_values = np.array([x, y])

        if self.car.car_created:
            self.car.updateStep(gps_values)

    def odom_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        velocity_rpm = (msg.lb_speed + msg.rb_speed) / 2
        velocity_mean = (velocity_rpm * 2 * math.pi * self.car.radio) / 60
        steering = msg.steering

        if self.car.car_created:
            self.car.updateStepVel(velocity_mean)
            self.car.updateStepSteer(time, steering)

    def gps_vel_callback(self, msg):
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        if self.car.car_created:
            self.car.updateStepVel(velocity_mean)

    def imu_callback(self, msg):
        imudata_linear = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        acceleration = math.sqrt(imudata_linear[0] ** 2 + imudata_linear[1] ** 2)

        if self.car.car_created:
            self.car.updateStepAcc(timestamp, acceleration)
        else:
            self.car.imu_past_time = timestamp

if __name__ == '__main__':
    rospy.init_node('EKF_node', anonymous=True)
    ekf_class = EKF_Class()
    rate = rospy.Rate(50)  # Frecuencia de los publishers (Hz)

    while not rospy.is_shutdown():
        ekf_class.path_pub.publish(ekf_class.car.path)
        ekf_class.pose_pub.publish(ekf_class.car.pose)
        # ekf_class.control_for_slam_pub.publish(ekf_class.car.publish_control())
        ekf_class.marker_array_pub.publish(ekf_class.car.marker_ests)
        rate.sleep()