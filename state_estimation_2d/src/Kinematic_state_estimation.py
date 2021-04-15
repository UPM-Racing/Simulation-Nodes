#! /usr/bin/env python
import rospy
import numpy as np
import math
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3Stamped
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from eufs_msgs.msg import WheelSpeedsStamped
from visualization_msgs.msg import MarkerArray, Marker
from ackermann_msgs.msg import AckermannDriveStamped

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

    return xEast, yNorth

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
        self.x = np.zeros(2)
        self.v = 0.0
        self.a = 0.0
        self.yaw = 0.0

        self.p_cov = np.eye(5)
        self.past_time = 0.0
        self.imu_past_time = 0.0

        self.var_imu_x = 0.10
        self.var_imu_v = 0.10
        self.var_imu_a = 0.10
        self.var_imu_w = 0.25
        self.var_imu_yaw = 0.10
        self.var_gnss = 0.25
        self.var_odom = 0.25

        self.h_jac = np.zeros([2, 5])
        self.h_jac[:, :2] = np.eye(2)  # measurement model jacobian

        self.h_vel_jac = np.zeros([1, 5])
        self.h_vel_jac[:, 2] = 1.0  # measurement model jacobian

        self.h_acc_jac = np.zeros([1, 5])
        self.h_acc_jac[:, 3] = 1.0  # measurement model jacobian

        self.h_steer_jac = np.zeros([1, 5])
        self.h_steer_jac[:, 4] = np.eye(1)  # measurement model jacobian

        self.car_created = False
        self.wheelbase = 1.58
        self.radio = 0.2525

        self.path = Path()
        self.pose = PoseStamped()
        self.control_msg = WheelSpeedsStamped()

    def InicializacionCoche(self, time):
        self.past_time = time
        self.car_created = True

    def Kinematic_prediction(self, time, u):
        # Update time measurement
        delta_t = time - self.past_time
        self.past_time = time

        lr = 0.711
        beta = math.atan2(lr * math.tan(u[1]), self.wheelbase)

        x_dot = self.v * math.cos(self.yaw + beta)
        y_dot = self.v * math.sin(self.yaw + beta)
        yaw_dot = self.v * math.sin(beta) / lr
        #yaw_dot = self.v * math.cos(beta) * math.tan(u[1]) / self.wheelbase

        self.a = u[0]
        self.x[0] = self.x[0] + x_dot * delta_t
        self.x[1] = self.x[1] + y_dot * delta_t
        self.yaw = self.yaw + yaw_dot * delta_t
        self.yaw = self.pi_2_pi(self.yaw)
        self.v = self.v + self.a * delta_t

        F = np.array([[1, 0, math.cos(self.yaw) * delta_t, 0, -self.v * math.sin(self.yaw) * delta_t],
                      [0, 1, math.sin(self.yaw) * delta_t, 0, self.v * math.cos(self.yaw) * delta_t],
                      [0, 0, 1, delta_t, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, (math.sin(beta) / lr) * delta_t, 0, 1]])

        Q = np.eye(5)
        Q[0:2, 0:2] = self.var_imu_x * Q[0:2, 0:2]
        Q[2:3, 2:3] = self.var_imu_v * Q[2:3, 2:3]
        Q[3:4, 3:4] = self.var_imu_a * Q[3:4, 3:4]
        Q[4:5, 4:5] = self.var_imu_w * Q[4:5, 4:5]
        Q = (delta_t ** 2) * Q

        # Propagate uncertainty
        self.p_cov = F.dot(self.p_cov).dot(F.T) + Q

        self.publish_path()

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def updateStep(self, gps_coord):
        # 3.1 Compute Kalman Gain
        R = self.var_gnss * np.eye(2)
        K = self.p_cov.dot(self.h_jac.T.dot(np.linalg.inv(self.h_jac.dot(self.p_cov.dot(self.h_jac.T)) + R)))

        # 3.2 Compute error state
        delta_x = K.dot(gps_coord - self.x)

        # 3.3 Correct predicted state
        self.x = self.x + delta_x[:2]
        self.v = self.v + delta_x[2]
        self.a = self.a + delta_x[3]
        self.yaw = self.yaw + delta_x[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_jac)).dot(self.p_cov)

    def updateStepVel(self, velocity):
        # 3.1 Compute Kalman Gain
        R = self.var_odom * np.eye(1)
        K = self.p_cov.dot(self.h_vel_jac.T.dot(np.linalg.inv(self.h_vel_jac.dot(self.p_cov.dot(self.h_vel_jac.T)) + R)))

        # 3.2 Compute error state
        delta_v = K.dot(np.array([velocity - self.v]))

        # 3.3 Correct predicted state
        self.x = self.x + delta_v[:2]
        self.v = self.v + delta_v[2]
        self.a = self.a + delta_v[3]
        self.yaw = self.yaw + delta_v[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_vel_jac)).dot(self.p_cov)

    def updateStepAcc(self,time, acceleration):
        delta_t = time - self.imu_past_time
        self.imu_past_time = time

        # 3.1 Compute Kalman Gain
        R = self.var_imu_a * np.eye(1)
        K = self.p_cov.dot(self.h_vel_jac.T.dot(np.linalg.inv(self.h_vel_jac.dot(self.p_cov.dot(self.h_vel_jac.T)) + R)))

        # 3.2 Compute error state
        velocity = self.v + delta_t * acceleration
        delta_v = K.dot(np.array([velocity - self.v]))

        # 3.3 Correct predicted state
        self.x = self.x + delta_v[:2]
        self.v = self.v + delta_v[2]
        self.a = self.a + delta_v[3]
        self.yaw = self.yaw + delta_v[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_vel_jac)).dot(self.p_cov)

    def updateStepSteer(self, time, steering):
        # Update time measurement
        delta_t = time - self.odom_past_time
        self.odom_past_time = time

        # 3.1 Compute Kalman Gain
        R = self.var_odom * np.eye(1)
        K = self.p_cov.dot(self.h_steer_jac.T.dot(np.linalg.inv(self.h_steer_jac.dot(self.p_cov.dot(self.h_steer_jac.T)) + R)))

        # 3.2 Compute error state
        self.odom_q_est = self.odom_q_est + delta_t * steering
        delta_q = K.dot(np.array([self.odom_q_est - self.q_est]))

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_q[:2]
        self.v_est = self.v_est + delta_q[2:4]
        self.q_est = self.q_est + delta_q[4]
        self.odom_q_est = self.q_est

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_steer_jac)).dot(self.p_cov)

        return self.p_est, self.v_est, self.q_est

    def updateAngleQuat(self, orientation):
        # 3.1 Compute Kalman Gain
        R = self.var_imu_yaw * np.eye(1)
        K = self.p_cov.dot(self.h_steer_jac.T.dot(np.linalg.inv(self.h_steer_jac.dot(self.p_cov.dot(self.h_steer_jac.T)) + R)))

        # 3.2 Compute error state
        yaw = np.arctan2(2 * (orientation.w * orientation.z), 1 - 2 * (orientation.z ** 2))
        delta_q = K.dot(np.array([yaw - self.q_est]))

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_q[:2]
        self.v_est = self.v_est + delta_q[2:4]
        self.q_est = self.q_est + delta_q[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_steer_jac)).dot(self.p_cov)

    def updateAngle(self, yaw):
        # 3.1 Compute Kalman Gain
        R = self.var_imu_yaw * np.eye(1)
        K = self.p_cov.dot(
            self.h_steer_jac.T.dot(np.linalg.inv(self.h_steer_jac.dot(self.p_cov.dot(self.h_steer_jac.T)) + R)))

        # 3.2 Compute error state
        delta_q = K.dot(np.array([yaw - self.q_est]))

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_q[:2]
        self.v_est = self.v_est + delta_q[2:4]
        self.q_est = self.q_est + delta_q[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_steer_jac)).dot(self.p_cov)

    def publish_path(self):
        self.path.header.stamp = rospy.Time.now()
        self.path.header.frame_id = "map"
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        point = Point()
        point.x = self.x[0]
        point.y = self.x[1]
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

    def publish_control(self):
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.header.frame_id = "map"
        self.control_msg.rf_speed = self.v

        return self.control_msg

class EKF_Class(object):
    def __init__(self):
        #self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        #self.imu_angle_sub = rospy.Subscriber('/imu', Imu, self.imu_angle_callback)

        self.gps = Gps()
        self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)
        self.path_pub = rospy.Publisher('/path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/pose_pub', PoseStamped, queue_size=1)
        self.control_for_slam_pub = rospy.Publisher('/control_for_slam', WheelSpeedsStamped, queue_size=1)

        #self.odom_sub = rospy.Subscriber('/ros_can/wheel_speeds', WheelSpeedsStamped, self.odom_callback)
        #self.odom_pub = rospy.Publisher('/odom_pub', WheelSpeedsStamped, queue_size=1)

        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)

        #self.marker_array_pub = rospy.Publisher('/marker_array_pub', MarkerArray, queue_size=10)

        self.car = Car()
        self.first = 2
        self.control_steering = 0.0

    def gps_callback(self, msg):
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        x, y = self.gps.gps_loop(latitude, longitude, altitude)
        gps_values = np.array([x, y])

        if self.car.car_created:
            self.car.updateStep(gps_values)

    def imu_callback(self, msg):
        imudata_angular = msg.angular_velocity.z
        imudata_linear = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        #orientation = msg.orientation
        acceleration = math.sqrt(imudata_linear[0] ** 2 + imudata_linear[1] ** 2)

        if self.car.car_created:
            #self.car.updateAngleQuat(orientation)
            self.car.updateStepAcc(timestamp, acceleration)
        else:
            self.car.imu_past_time = timestamp

    def imu_angle_callback(self, msg):
        if self.car.car_created:
            self.car.updateAngleQuat(msg.orientation)

    def odom_callback(self, msg):
        odometry = msg
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_rpm = (msg.lb_speed + msg.rb_speed) / 2
        velocity_mean = (velocity_rpm * 2 * math.pi * self.car.radio) / 60

        # Max steering = 0.52 rad/s
        steering = msg.steering

        if self.car.car_created:
            #self.car.updateStepVel(timestamp, velocity)
            #self.car.updateStepSteer(timestamp, steering)

            odometry.lf_speed = steering
            odometry.rf_speed = self.control_steering
            odometry.lb_speed = self.control_steering - steering
            #self.odom_pub.publish(odometry)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        yaw = np.arctan2(velocity[1], velocity[0])

        if self.car.car_created:
            self.car.updateStepVel(velocity_mean)

    def control_callback(self, msg):
        u = np.zeros(2)
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration
        u[0] = acceleration
        u[1] = angle
        self.control_steering = angle
        self.car.control_msg.steering = angle

        if self.first != 0:
            self.car.InicializacionCoche(time)
            self.first -= 1
        self.car.Kinematic_prediction(time, u)

    def append_pose(self, p_est, q_est, msg):
        pose = PoseStamped()
        pose.header = msg.header
        point = Point()
        point.x = p_est[0]
        point.y = p_est[1]
        point.z = 0.25
        pose.pose.position = point
        orientation = Quaternion()
        # Suponiendo roll y pitch = 0
        orientation.x = 0.0
        orientation.y = 0.0
        orientation.z = np.sin(q_est * 0.5)
        orientation.w = np.cos(q_est * 0.5)
        pose.pose.orientation = orientation

        return pose

    def plot_covariance_ellipse(self, p_est, p_cov, header):  # pragma: no cover
        Pxy = p_cov[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.2)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
        C_ns = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        fx = np.matmul(C_ns, np.array([x, y]))
        px = np.array(fx[0, :] + p_est[0]).flatten()
        py = np.array(fx[1, :] + p_est[1]).flatten()

        self.marker_array(px, py, header)

    def marker_array(self, px, py, header):
        # Publish it as a marker in rviz
        marker_ests = MarkerArray()
        marker_ests.markers = []
        for i in range(len(px)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CUBE
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = px[i]
            point.y = py[i]
            point.z = 0.0
            pose.position = point
            orientation = Quaternion()
            # Suponiendo roll y pitch = 0
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            pose.orientation = orientation
            marker_est.pose = pose
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (255, 0, 0)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.01, 0.01, 0.01)
            marker_ests.markers.append(marker_est)

        #self.marker_array_pub.publish(marker_ests)

if __name__ == '__main__':
    rospy.init_node('EKF_node', anonymous=True)
    ekf_class = EKF_Class()
    #rospy.spin()
    rate = rospy.Rate(50)  # Hz

    while not rospy.is_shutdown():
        ekf_class.path_pub.publish(ekf_class.car.path)
        ekf_class.pose_pub.publish(ekf_class.car.pose)
        ekf_class.control_for_slam_pub.publish(ekf_class.car.publish_control())
        rate.sleep()
