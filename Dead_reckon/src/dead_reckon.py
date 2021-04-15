#! /usr/bin/env python
import rospy
import math
import numpy as np
from eufs_msgs.msg import ConeArrayWithCovariance, CarState
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion, Vector3Stamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from eufs_msgs.msg import WheelSpeedsStamped
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped

class Dead_reckon:
    def __init__(self):
        self.past_time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.DT = 0.0
        self.wheelbase = 1.58
        self.marker_ests = MarkerArray()
        self.u = np.zeros((2, 1))
        self.path = Path()
        self.pose = PoseStamped()
        self.R = np.diag([1.0, np.deg2rad(10.0)]) ** 2

    def bucle_principal(self, time, yaw):
        self.DT = time - self.past_time
        self.past_time = time

        self.u[1, 0] = yaw

        px = np.zeros((3, 1))
        px[0, 0] = self.x
        px[1, 0] = self.y
        px[2, 0] = self.yaw
        # se vuelve a anadir ruido a el control input vector
        # (2,1) u = [vel, yaw_rate]
        ud = self.u + (np.matmul(np.random.randn(1, 2), self.R ** 0.5)).T  # add noise
        # se actualiza el estado de las particulas en funcion del motion model
        px = self.motion_model(px, self.u)
        self.x = px[0, 0]
        self.y = px[1, 0]
        self.yaw = px[2, 0]

        print('-------------------')
        print(self.DT)
        print(self.yaw)
        print(yaw)

        self.publish_path()

    def motion_model(self, x, u):

        x[2, 0] = x[2, 0] + u[1, 0] * self.DT
        x[0, 0] = x[0, 0] + self.DT * math.cos(x[2, 0]) * u[0, 0]
        x[1, 0] = x[1, 0] + self.DT * math.sin(x[2, 0]) * u[0, 0]

        x[2, 0] = self.pi_2_pi(x[2, 0])

        return x

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def publish_path(self):
        self.path.header.stamp = rospy.Time.now()
        self.path.header.frame_id = "map"
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        point = Point()
        point.x = self.x
        point.y = self.y
        point.z = 0.25
        pose.pose.position = point
        orientation = Quaternion()
        # Suponiendo roll y pitch = 0
        orientation.x = 0.0
        orientation.y = 0.0
        orientation.z = np.sin(self.yaw * 0.5)
        orientation.w = np.cos(self.yaw * 0.5)
        pose.pose.orientation = orientation
        self.path.poses.append(pose)
        self.pose = pose

    def update_car_position(self, pose):
        self.x = pose.position.x
        self.y = pose.position.y
        self.yaw = self.pi_2_pi(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))

    def update_car_position_odom(self, time, velocity, yaw_rate):
        self.u[0] = velocity
        self.u[1] = yaw_rate

    def update_car_yaw(self, yaw):
        self.u[1] = yaw

    def update_car_gps_vel(self, velocity):
        self.u[0, 0] = velocity


class Slam_Class(object):
    def __init__(self):
        self.dead_reckon = Dead_reckon()

        # Solo pueden estar conectado 1 de estos dos siguientes
        #self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        # self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.state_estimation_callback)

        # Solo pueden estar conectados odom, o imu y gps
        # self.odom_sub = rospy.Subscriber('/odometry_pub', WheelSpeedsStamped, self.odom_callback)
        # self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)
        self.first = 2

        self.path_pub = rospy.Publisher('/dead_reckon_path', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/dead_reckon_pose', PoseStamped, queue_size=1)

    def sub_callback(self, msg):
        self.dead_reckon.update_car_position(msg.pose.pose)

    def state_estimation_callback(self, msg):
        self.dead_reckon.update_car_position(msg.pose)

    def odom_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_rpm = msg.lb_speed
        velocity_mean = (velocity_rpm * 2.0 * math.pi * 0.25) / 60.0

        # Max steering = 0.52 rad/s
        steering = msg.steering

        self.dead_reckon.update_car_position_odom(timestamp, velocity_mean, steering)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        self.dead_reckon.update_car_gps_vel(velocity_mean)

    def imu_callback(self, msg):
        yaw_rate = msg.angular_velocity.z
        self.dead_reckon.update_car_yaw(yaw_rate)

    def control_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration

        if self.first != 0:
            self.dead_reckon.past_time = time
            self.first -= 1
        self.dead_reckon.bucle_principal(time, angle)


if __name__ == '__main__':
    rospy.init_node('slam_node', anonymous=True)
    slam_class = Slam_Class()
    #rospy.spin()
    rate = rospy.Rate(10)  # Hz

    while not rospy.is_shutdown():
        slam_class.path_pub.publish(slam_class.dead_reckon.path)
        slam_class.pose_pub.publish(slam_class.dead_reckon.pose)
        rate.sleep()
