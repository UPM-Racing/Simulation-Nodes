#! /usr/bin/env python

import numpy as np
import rospy
import math
import matplotlib.pyplot as plt
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from nav_msgs.msg import Path
import scipy.interpolate as scipy_interpolate
from eufs_msgs.msg import CarState, CanState

Kp = 1.0                                    # speed proportional gain
ki = 1.0                                    # speed integral gain
kd = 0.1                                    # speed derivational gain
dt = 0.1                                    # [s] time difference
target_speed = 10.0 / 3.6                   # [m/s]
Ke = 5                                      # control gain
Kv = 1
max_steer = 27.2 * np.pi / 180              # [rad] max steering angle
max_accel = 1.0                             # [m/s^2] max acceleration


class Stanley(object):
    def __init__(self):
        self.v = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.n_course_point = 200
        self.degree = 3

        self.contador = 0
        self.mean_deviation = 0.0
        self.error = []

        self.ack_msg = AckermannDriveStamped()
        self.ack_msg.header.frame_id = "map"
        self.ack_msg.drive.speed = 0                    # [m/s]
        self.ack_msg.drive.steering_angle = 0.0         # [rad]
        self.ack_msg.drive.steering_angle_velocity = 0  # [rad/s]
        self.ack_msg.drive.acceleration = 0.0           # [m/s^2]
        self.ack_msg.drive.jerk = 0                     # [m/s^3]

    def principal_loop(self, path_planning):
        poses = path_planning.poses
        positions_x = []
        positions_y = []

        for i, pose in enumerate(poses):
            positions_x.append(pose.pose.position.x)
            positions_y.append(pose.pose.position.y)

        if len(positions_x) > self.degree:
            path_x, path_y = self.interpolate_b_spline_path(positions_x, positions_y, self.n_course_point, self.degree)
        elif len(positions_x) >= 2:
            path_x, path_y = self.interpolate_b_spline_path(positions_x, positions_y, self.n_course_point, len(positions_x) - 1)

        if 'path_x' in locals():
            self.ack_msg.header.stamp = rospy.Time.now()
            self.ack_msg.drive.steering_angle = self.stanley_control(path_x, path_y)
            self.ack_msg.drive.acceleration = self.pid_control(target_speed)

    def interpolate_b_spline_path(self, x, y, n_path_points, degree):
        """
        interpolate points with a B-Spline path
        :param x: x positions of interpolated points
        :param y: y positions of interpolated points
        :param n_path_points: number of path points
        :param degree: B-Spline degree
        :return: x and y position list of the result path
        """
        ipl_t = np.linspace(0.0, len(x) - 1, len(x))
        spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
        spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)

        travel = np.linspace(0.0, len(x) - 1, n_path_points)
        return spl_i_x(travel), spl_i_y(travel)

    def update_car_position(self, pose):
        self.x = pose.position.x
        self.y = pose.position.y
        self.yaw = self.normalize_angle(
            np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))

    def update_velocity(self, v):
        self.v = v

    def pid_control(self, target):
        accel = Kp * (target - self.v) + ki * (target - self.v) * dt + kd * (
                target - self.v) / dt
        accel = np.clip(accel, 0, max_accel)        # Freno por inercia, maxima aceleracion 1 m/s^2

        # print("Aceleracion: {}".format(accel))

        return accel

    def stanley_control(self, path_x, path_y):
        self.contador = self.contador + 1
        current_target_idx = self.calc_target_spline(path_x, path_y)
        # Yaw path calculation through the tangent of two points
        if current_target_idx < (len(path_x) - 1):
            diff_x = path_x[current_target_idx + 1] - path_x[current_target_idx]
            diff_y = path_y[current_target_idx + 1] - path_y[current_target_idx]
        else:
            diff_x = path_x[current_target_idx] - path_x[current_target_idx - 1]
            diff_y = path_y[current_target_idx] - path_y[current_target_idx - 1]
        yaw_path = np.arctan2(diff_y, diff_x)

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(yaw_path - self.yaw)

        # theta_d corrects the cross track error
        # A, B, C represent the general equation of a line that passes through two points
        if current_target_idx < (len(path_x) - 1):
            A = path_y[current_target_idx + 1] - path_y[current_target_idx]
            B = path_x[current_target_idx] - path_x[current_target_idx + 1]
            C = path_y[current_target_idx] * path_x[current_target_idx + 1] - path_x[current_target_idx] * \
                path_y[current_target_idx + 1]
        else:
            A = path_y[current_target_idx] - path_y[current_target_idx - 1]
            B = path_x[current_target_idx - 1] - path_x[current_target_idx]
            C = path_y[current_target_idx - 1] * path_x[current_target_idx] - path_x[current_target_idx - 1] * \
                path_y[current_target_idx]

        error_front_axle = (A * self.x + B * self.y + C) / np.sqrt(
            A ** 2 + B ** 2)  # Distance from a point to a line in the plane
        theta_d = np.arctan2(Ke * error_front_axle, Kv + self.v)

        # Steering control
        delta = theta_e + theta_d
        delta = np.clip(delta, -max_steer, max_steer)

        self.error.append(error_front_axle)

        # Calculate mean deviation (MD = (1/n)*sum(error))
        self.mean_deviation = (self.mean_deviation * (self.contador - 1) + abs(error_front_axle)) * (self.contador ** (-1))

        # print("Mean deviation: {}".format(self.mean_deviation))

        return delta

    def calc_target_spline(self, path_x, path_y):
        """
        Calculate the index of the closest point of the path
        :param path_x: [float] x coordinates list of the path planning
        :param path_y: [float] y coordinates list of the path planning
        :return: (int, float)  Index of the way point at the shortest distance
        """
        min_idx = 0
        min_dist = float("inf")

        # Search nearest waypoint
        for i in range(len(path_x)):
            dist = np.linalg.norm(np.array([path_x[i] - self.x, path_y[i] - self.y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi


class State(object):
    def __init__(self):
        self.stanley_class = Stanley()
        self.gps_vel = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.callbackgps)
        self.path_planning_sub = rospy.Subscriber('/path_planning_pub', Path, self.callbackpath)
        # self.real_path = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        # self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.sub_callback2)
        self.state_estimation_sub = rospy.Subscriber('/slam_pose_pub', PoseStamped, self.sub_callback2)
        self.state_sub = rospy.Subscriber('/ros_can/state', CanState, self.callbackstate)

        self.control = rospy.Publisher('/cmd_vel_out', AckermannDriveStamped, queue_size=1)
        self.start = rospy.Publisher('/ros_can/mission_flag', Bool, queue_size=1)

        self.state = False

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
        self.stanley_class.principal_loop(data)

    def callbackstate(self, msg):
        if msg.as_state == 2:
            self.state = True


if __name__ == '__main__':
    rospy.init_node('stanley', anonymous=True)
    state = State()
    rate = rospy.Rate(200)    # Hz
    show_animation = False   # To decide if we want to graph the error

    while not rospy.is_shutdown():
        if state.state:
            state.control.publish(state.stanley_class.ack_msg)
        rate.sleep()

    if show_animation:  # pragma: no cover
        t = np.arange(0, len(state.stanley_class.error), 1)
        plt.figure()
        plt.ylim(-1.0, 1.0)
        plt.plot(t, state.stanley_class.error, ".r", alpha=.5)
        plt.xlabel("Time")
        plt.ylabel("Position Error")
        plt.grid(True)
        plt.show()

    target_speed = 0.0  # Braking the car
