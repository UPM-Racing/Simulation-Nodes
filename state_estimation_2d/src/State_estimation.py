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
        self.p_est = np.zeros(2)
        self.v_est = np.zeros(2)
        self.q_est = 0.0
        self.p_cov = np.eye(5)
        self.past_time = 0.0
        self.odom_past_time = 0.0
        self.odom_q_est = 0.0

        self.var_imu_x = 0.10
        self.var_imu_v = 0.10
        self.var_imu_w = 0.25
        self.var_imu_yaw = 0.10
        self.var_gnss = 0.50
        self.var_odom = 0.25

        self.h_jac = np.zeros([2, 5])
        self.h_jac[:, :2] = np.eye(2)  # measurement model jacobian

        self.h_vel_jac = np.zeros([2, 5])
        self.h_vel_jac[:, 2:4] = np.eye(2)  # measurement model jacobian

        self.h_steer_jac = np.zeros([1, 5])
        self.h_steer_jac[:, 4] = np.eye(1)  # measurement model jacobian

        self.car_created = False

    def InicializacionCoche(self, time):
        self.past_time = time
        self.odom_past_time = time
        self.car_created = True

    def predictionStep(self, time, w_vect, f_vect):
        # Update time measurement
        delta_t = time - self.past_time
        self.past_time = time

        # Update state with IMU inputs
        C_ns = np.array([[np.cos(self.q_est), -np.sin(self.q_est)],
                         [np.sin(self.q_est), np.cos(self.q_est)]])

        self.p_est = self.p_est + delta_t * self.v_est + 0.5 * (delta_t ** 2) * C_ns.dot(f_vect)
        self.v_est = self.v_est + delta_t * C_ns.dot(f_vect)
        self.q_est = self.q_est + delta_t * w_vect

        # Linearize Motion Model
        F = np.eye(5)
        F[0:2, 2:4] = delta_t * np.eye(2)

        Q = np.eye(5)
        Q[0:2, 0:2] = self.var_imu_x * Q[0:2, 0:2]
        Q[2:4, 2:4] = self.var_imu_v * Q[2:4, 2:4]
        Q[4:5, 4:5] = self.var_imu_w * Q[4:5, 4:5]
        Q = (delta_t ** 2) * Q

        # Propagate uncertainty
        self.p_cov = F.dot(self.p_cov).dot(F.T) + Q

        return self.p_est, self.q_est

    def updateStep(self, time, gps_coord):
        # 3.1 Compute Kalman Gain
        R = self.var_gnss * np.eye(2)
        K = self.p_cov.dot(self.h_jac.T.dot(np.linalg.inv(self.h_jac.dot(self.p_cov.dot(self.h_jac.T)) + R)))

        # 3.2 Compute error state
        delta_x = K.dot(gps_coord - self.p_est)

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_x[:2]
        self.v_est = self.v_est + delta_x[2:4]
        self.q_est = self.q_est + delta_x[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_jac)).dot(self.p_cov)

        return self.p_est, self.q_est

    def updateStepVel(self, time, velocity):
        # 3.1 Compute Kalman Gain
        R = self.var_odom * np.eye(2)
        K = self.p_cov.dot(self.h_vel_jac.T.dot(np.linalg.inv(self.h_vel_jac.dot(self.p_cov.dot(self.h_vel_jac.T)) + R)))

        # 3.2 Compute error state
        delta_v = K.dot(velocity - self.v_est)

        # 3.3 Correct predicted state
        self.p_est = self.p_est + delta_v[:2]
        self.v_est = self.v_est + delta_v[2:4]
        self.q_est = self.q_est + delta_v[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(self.h_vel_jac)).dot(self.p_cov)

        return self.p_est, self.v_est, self.q_est

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


class EKF_Class(object):
    def __init__(self):
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        #self.imu_angle_sub = rospy.Subscriber('/imu', Imu, self.imu_angle_callback)

        self.gps = Gps()
        self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)
        self.path_pub = rospy.Publisher('/path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/pose_pub', PoseStamped, queue_size=1)

        #self.odom_sub = rospy.Subscriber('/ros_can/wheel_speeds', WheelSpeedsStamped, self.odom_callback)
        #self.odom_pub = rospy.Publisher('/odom_pub', WheelSpeedsStamped, queue_size=10)

        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)

        #self.marker_array_pub = rospy.Publisher('/marker_array_pub', MarkerArray, queue_size=10)

        self.car = Car()
        self.path = Path()
        self.pose = PoseStamped()

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
            self.pose = gps_pose
            #self.plot_covariance_ellipse(p_est, self.car.p_cov, msg.header)

    def imu_callback(self, msg):
        imudata_angular = msg.angular_velocity.z
        imudata_linear = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y])
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        #orientation = msg.orientation

        if self.car.car_created:
            #self.car.updateAngleQuat(orientation)
            p_est, q_est = self.car.predictionStep(timestamp, imudata_angular, imudata_linear)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            imu_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(imu_pose)
            self.pose = imu_pose
            #self.plot_covariance_ellipse(p_est, self.car.p_cov, msg.header)
        else:
            self.car.InicializacionCoche(timestamp)

    def imu_angle_callback(self, msg):
        if self.car.car_created:
            self.car.updateAngleQuat(msg.orientation)

    def odom_callback(self, msg):
        odometry = msg
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_rpm = (msg.lb_speed + msg.rb_speed) / 2
        velocity_mean = (velocity_rpm * 2 * math.pi * 0.25) / 60

        C_ns = np.array([[np.cos(self.car.q_est), -np.sin(self.car.q_est)],
                         [np.sin(self.car.q_est), np.cos(self.car.q_est)]])

        velocity = C_ns.dot([velocity_mean, 0.0])

        # Max steering = 0.52 rad/s
        steering = msg.steering
        #print(velocity_rpm, velocity_mean)

        if self.car.car_created:
            #p_est, v_est, q_est = self.car.updateStepVel(timestamp, velocity)
            p_est, v_est, q_est = self.car.updateStepSteer(timestamp, steering)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            odom_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(odom_pose)
            self.pose = odom_pose
            #self.plot_covariance_ellipse(p_est, self.car.p_cov, msg.header)

            #odometry.lf_speed = v_est[0]
            #odometry.rf_speed = velocity_mean
            #self.odom_pub.publish(odometry)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = msg.header.stamp.nsecs / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        yaw = np.arctan2(velocity[1], velocity[0])
        #print(velocity, yaw)
        if self.car.car_created:
            #self.car.updateAngle(yaw)
            p_est, v_est, q_est = self.car.updateStepVel(timestamp, velocity)
            self.path.header = msg.header
            self.path.header.frame_id = "map"
            gps_vel_pose = self.append_pose(p_est, q_est, msg)
            self.path.poses.append(gps_vel_pose)
            self.pose = gps_vel_pose
            #self.plot_covariance_ellipse(p_est, self.car.p_cov, msg.header)

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

        self.marker_array_pub.publish(marker_ests)

if __name__ == '__main__':
    rospy.init_node('EKF_node', anonymous=True)
    ekf_class = EKF_Class()
    #rospy.spin()
    rate = rospy.Rate(50)  # Hz

    while not rospy.is_shutdown():
        ekf_class.path_pub.publish(ekf_class.path)
        ekf_class.pose_pub.publish(ekf_class.pose)
        rate.sleep()
