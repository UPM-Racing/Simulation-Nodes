#! /usr/bin/env python
import rospy
import numpy as np
import rotations as rot
#from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from eufs_msgs.msg import WheelSpeedsStamped

class Gps():

  def __init__(self):
    self.a = 6378137
    self.b = 6356752.3142
    self.f = (self.a - self.b) / self.a
    self.e_sq = self.f * (2 - self.f)

    self.LONGITUD_0 = 0
    self.LATITUD_0 = 0
    self.ALTURA_0 = 0

  def geodetic_to_ecef(self, lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = self.a / math.sqrt(1 - self.e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - self.e_sq) * N) * sin_lambda

    return x, y, z

  def ecef_to_enu(self, x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = self.a / math.sqrt(1 - self.e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - self.e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp

  def geodetic_to_enu(self, lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = self.geodetic_to_ecef(lat, lon, h)

    return self.ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

  def gps_loop(self, latitude, longitude, altitude):
    if self.LONGITUD_0 == 0:
      self.LONGITUD_0 = longitude
      self.LATITUD_0 = latitude
      self.ALTURA_0 = altitude

    gps_coordinates = self.geodetic_to_enu(latitude, longitude, altitude, self.LATITUD_0, self.LONGITUD_0, self.ALTURA_0)

    return gps_coordinates

# Class of the car definition
class Car():

    def __init__(self):
         # super(Car, self).__init__() No hay herencia
        self.p_est = np.zeros(3)
        self.v_est = np.zeros(3)
        self.q_est = rot.Quaternion().to_numpy()
        self.p_cov = np.eye(9)
        self.list_x = []
        self.list_y = []
        self.list_z = []
        self.past_time = 0.0

        self.var_imu_f = 0.50
        self.var_imu_w = 0.50
        self.var_gnss = 0.50
        self.var_odom = 0.25

        self.g = np.array([0, 0, -9.81])  # gravity
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian

        self.h_vel_jac = np.zeros([3, 9])
        self.h_vel_jac[:, 3:6] = np.eye(3)  # measurement model jacobian

        self.car_created = False

    def InicializacionCoche(self, time):
        self.past_time = time
        #self.q_est = rot.Quaternion(cuaternio).to_numpy()
        self.car_created = True

    def storeValues(self):
        self.list_x.append(self.p_est[0])
        self.list_y.append(self.p_est[1])
        self.list_z.append(self.p_est[2])

    def updateEuler(self, time, euler):
        self.q_est = rot.Quaternion(euler=euler).to_numpy()

    def updateCuaternio(self, time, cuaternio):
        self.q_est = rot.Quaternion(cuaternio).to_numpy()

    def predictionStep(self, time, w_vect, f_vect):
        # Update time measurement
        delta_t = time - self.past_time
        self.past_time = time

        # Update state with IMU inputs
        C_ns = rot.Quaternion(*self.q_est).to_mat()
        self.p_est = self.p_est + delta_t * self.v_est + 0.5 * (delta_t ** 2) * (C_ns.dot(f_vect) - self.g)
        self.v_est = self.v_est + delta_t * (C_ns.dot(f_vect) - self.g)
        self.q_est = rot.Quaternion(euler=(delta_t * w_vect)).quat_mult_right(self.q_est)

        # Linearize Motion Model
        F = np.eye(9)
        imu = f_vect.reshape((3, 1))
        F[0:3, 3:6] = delta_t * np.eye(3)
        F[3:6, 6:9] = - rot.skew_symmetric(C_ns.dot(imu)) * delta_t

        Q = np.eye(6)
        Q[0:3, 0:3] = self.var_imu_f * Q[0:3, 0:3]
        Q[3:6, 3:6] = self.var_imu_w * Q[3:6, 3:6]
        Q = (delta_t ** 2) * Q

        # Propagate uncertainty
        self.p_cov = F.dot(self.p_cov).dot(F.T) + self.l_jac.dot(Q).dot(self.l_jac.T)

        # Store values
        #self.storeValues()
        return self.p_est, self.q_est

    def updateStep(self, time, gps_coord):
        # 3.1 Compute Kalman Gain
        R = self.var_gnss * np.eye(3)
        K = self.p_cov.dot(self.h_jac.T.dot(np.linalg.inv(self.h_jac.dot(self.p_cov.dot(self.h_jac.T)) + R)))

        # 3.2 Compute error state
        delta_x = K.dot(gps_coord - self.p_est)

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_x[:3]
        self.v_est = self.v_est + delta_x[3:6]
        self.q_est = rot.Quaternion(axis_angle=delta_x[6:]).quat_mult_left(self.q_est)

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(9) - K.dot(self.h_jac)).dot(self.p_cov)

        # Store values
        #self.storeValues()
        return self.p_est, self.q_est

    def updateStep2(self, time, velocity):
        # 3.1 Compute Kalman Gain
        R = self.var_odom * np.eye(3)
        K = self.p_cov.dot(self.h_vel_jac.T.dot(np.linalg.inv(self.h_vel_jac.dot(self.p_cov.dot(self.h_vel_jac.T)) + R)))

        # 3.2 Compute error state
        delta_v = K.dot(velocity - self.v_est)

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_v[:3]
        self.v_est = self.v_est + delta_v[3:6]
        self.q_est = rot.Quaternion(axis_angle=delta_v[6:]).quat_mult_left(self.q_est)

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(9) - K.dot(self.h_vel_jac)).dot(self.p_cov)

        # Store values
        #self.storeValues()
        return self.p_est, self.v_est, self.q_est


class EKF_Class(object):
    def __init__(self):
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        self.gps = Gps()
        self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)
        self.path_pub = rospy.Publisher('path_pub', Path, queue_size=10)
        self.pose_pub = rospy.Publisher('pose_pub', PoseStamped, queue_size=10)

        #self.odom_sub = rospy.Subscriber('/ros_can/wheel_speeds', WheelSpeedsStamped, self.odom_callback)
        #self.odom_pub = rospy.Publisher('odom_pub', WheelSpeedsStamped, queue_size=10)

        self.odom_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)

        self.car = Car()
        self.path = Path()

    def gps_callback(self, msg):
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        gps_values = self.gps.gps_loop(latitude, longitude, altitude)

        if self.car.car_created:
            p_est, q_est = self.car.updateStep(timestamp, gps_values)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            gps_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(gps_pose)
            self.path_pub.publish(self.path)
            self.pose_pub.publish(gps_pose)

    def imu_callback(self, msg):
        imudata_angular = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        imudata_linear = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec

        if self.car.car_created:
            p_est, q_est = self.car.predictionStep(timestamp, imudata_angular, imudata_linear)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            imu_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(imu_pose)
            self.path_pub.publish(self.path)
            self.pose_pub.publish(imu_pose)
        else:
            self.car.InicializacionCoche(timestamp)

    def odom_callback(self, msg):
        odometry = msg
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_mean = (msg.lb_speed + msg.rb_speed) / 2
        velocity = (velocity_mean * 2 * math.pi * 0.25) / 60

        if self.car.car_created:
            p_est, v_est, q_est = self.car.updateStep2(timestamp, [velocity, 0.0, 0.0])
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            odom_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(odom_pose)
            self.path_pub.publish(self.path)
            self.pose_pub.publish(odom_pose)

            odometry.lf_speed = v_est[0]
            odometry.rf_speed = velocity
            self.odom_pub.publish(odometry)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y, msg.vector.x, msg.vector.z]

        if self.car.car_created:
            p_est, v_est, q_est = self.car.updateStep2(timestamp, velocity)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            gps_vel_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(gps_vel_pose)
            self.path_pub.publish(self.path)
            self.pose_pub.publish(gps_vel_pose)

    def append_pose(self, p_est, q_est, msg):
        pose = PoseStamped()
        pose.header = msg.header
        point = Point()
        point.x = p_est[0]
        point.y = p_est[1]
        point.z = p_est[2]
        pose.pose.position = point
        orientation = Quaternion()
        orientation.x = q_est[1]
        orientation.y = q_est[2]
        orientation.z = q_est[3]
        orientation.w = q_est[0]
        pose.pose.orientation = orientation

        return pose

if __name__ == '__main__':
    rospy.init_node('EKF_node', anonymous=True)
    sensor_class = EKF_Class()
    rospy.spin()
