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

class EKFSLAM:
    def __init__(self):
        self.past_time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.DT = 0.0
        self.THRESHOLD = 1.0
        self.cones_x = []
        self.cones_y = []
        self.marker_ests = MarkerArray()
        self.total_landmarks = 0
        self.u = np.zeros((2, 1))
        self.path = Path()
        self.pose = PoseStamped()
        self.wheelbase = 1.58

        # Constants
        # EKF state covariance
        self.Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

        self.M_DIST_TH = 0.3  # Threshold of Mahalanobis distance for data association.
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]

        # State Vector [x y yaw v]'
        self.xEst = np.zeros((self.STATE_SIZE, 1))
        self.PEst = np.eye(self.STATE_SIZE)

    def bucle_principal(self, cones, time):
        self.DT = time - self.past_time
        self.past_time = time

        z = np.zeros((0, 2))

        for i, cone in enumerate(cones):
            d, angle = self.definir_posicion(cone.point.x, cone.point.y)
            zi = np.array([d, angle])
            z = np.vstack((z, zi))
            #print(cone.point.x, cone.point.y)

        #print(self.u)

        self.xEst, self.PEst = self.ekf_slam(self.xEst, self.PEst, self.u, z)

        #print(self.xEst)
        #print('----------------------')

        self.x = self.xEst[0]
        self.y = self.xEst[1]
        self.yaw = self.xEst[2]
        #print('yaw: ' + str(self.yaw * 360 / (2 * math.pi)))
        #print('u' + str(self.u))

        self.cones_x = []
        self.cones_y = []

        #print(self.calc_n_lm(self.xEst))

        # visible landmark
        for i in range(self.calc_n_lm(self.xEst)):
            self.cones_x.append(self.xEst[3 + i * 2])
            self.cones_y.append(self.xEst[3 + i * 2 + 1])

        #self.marker_array_slam()
        self.marker_array(self.cones_x, self.cones_y)
        self.publish_path()

    def ekf_slam(self, xEst, PEst, u, z):
        # Predict
        S = self.STATE_SIZE
        G, Fx = self.jacob_motion(xEst[0:S], u)
        xEst[0:S] = self.motion_model(xEst[0:S], u)
        PEst[0:S, 0:S] = np.matmul(np.matmul(G.T, PEst[0:S, 0:S]), G) + np.matmul(np.matmul(Fx.T, self.Cx), Fx)
        initP = np.eye(2)

        # Update
        for iz in range(len(z[:, 0])):  # for each observation
            min_id = self.search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

            nLM = self.calc_n_lm(xEst)
            if min_id == nLM:
                # Extend state and covariance matrix
                # Pasa a globales las coordenadas de las observaciones z y las anade a xEst
                xAug = np.vstack((xEst, self.calc_landmark_position(xEst, z[iz, :])))
                PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), self.LM_SIZE)))),
                                  np.hstack((np.zeros((self.LM_SIZE, len(xEst))), initP))))
                xEst = xAug
                PEst = PAug
                #print(nLM)
            lm = self.get_landmark_position_from_state(xEst, min_id)
            y, S, H = self.calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

            K = np.matmul(np.matmul(PEst, H.T), np.linalg.inv(S))
            xEst = xEst + np.matmul(K, y)
            PEst = np.matmul(np.eye(len(xEst)) - np.matmul(K, H), PEst)

        xEst[2] = self.pi_2_pi(xEst[2])

        return xEst, PEst

    def definir_posicion(self, x, y):
        d = math.hypot(x, y)
        angle = self.pi_2_pi(math.atan2(y, x))
        return d, angle

    def motion_model(self, x, u):
        lr = 0.711
        beta = math.atan2(lr * math.tan(u[1, 0]), self.wheelbase)

        x_dot = u[0, 0] * math.cos(x[2, 0] + beta)
        y_dot = u[0, 0] * math.sin(x[2, 0] + beta)
        yaw_dot = u[0, 0] * math.sin(beta) / lr

        x[0, 0] = x[0, 0] + x_dot * self.DT
        x[1, 0] = x[1, 0] + y_dot * self.DT
        x[2, 0] = x[2, 0] + yaw_dot * self.DT

        x[2, 0] = self.pi_2_pi(x[2, 0])

        return x

    def calc_n_lm(self, x):
        n = int((len(x) - self.STATE_SIZE) / self.LM_SIZE)
        return n

    def jacob_motion(self, x, u):
        Fx = np.hstack((np.eye(self.STATE_SIZE), np.zeros((self.STATE_SIZE, self.LM_SIZE * self.calc_n_lm(x)))))

        jF = np.array([[0.0, 0.0, -self.DT * u[0, 0] * math.sin(x[2, 0])],
                       [0.0, 0.0, self.DT * u[0, 0] * math.cos(x[2, 0])],
                       [0.0, 0.0, 0.0]], dtype=float)

        G = np.eye(self.STATE_SIZE) + np.matmul(np.matmul(Fx.T, jF), Fx)

        return G, Fx,

    def calc_landmark_position(self, x, z):
        zp = np.zeros((2, 1))

        #print(x[0:3, 0])

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

        # print(zp)
        # print('-------------------')

        return zp

    def get_landmark_position_from_state(self, x, ind):
        lm = x[self.STATE_SIZE + self.LM_SIZE * ind: self.STATE_SIZE + self.LM_SIZE * (ind + 1), :]

        return lm

    def search_correspond_landmark_id(self, xEst, PEst, zi):
        """
        Landmark association with Mahalanobis distance
        :param xEst: Vector de estado y de posicion de los landmarks
        :param PEst: Matriz de covarianzas del vector de estado y posicion de los landmarks
        :param zi: Posicion local de las observaciones (d, angle)
        :return: id del landmark que cumpla un threshold de distancia o anade un id nuevo
        """

        nLM = self.calc_n_lm(xEst)

        min_dist = []

        for i in range(nLM):
            lm = self.get_landmark_position_from_state(xEst, i)
            y, S, H = self.calc_innovation(lm, xEst, PEst, zi, i)
            min_dist.append(np.matmul(np.matmul(y.T, np.linalg.inv(S)), y))

        min_dist.append(self.M_DIST_TH)  # new landmark

        min_id = min_dist.index(min(min_dist))

        return min_id

    def calc_innovation(self, lm, xEst, PEst, z, LMid):
        delta = lm - xEst[0:2]
        q = (np.matmul(delta.T, delta))[0, 0]
        z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
        zp = np.array([[math.sqrt(q), self.pi_2_pi(z_angle)]])
        y = (z - zp).T
        y[1] = self.pi_2_pi(y[1])
        H = self.jacob_h(q, delta, xEst, LMid + 1)
        S = np.matmul(np.matmul(H, PEst), H.T) + self.Cx[0:2, 0:2]

        return y, S, H

    def jacob_h(self, q, delta, x, i):
        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                      [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

        G = G / q
        nLM = self.calc_n_lm(x)
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                        np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

        F = np.vstack((F1, F2))

        H = np.matmul(G, F)

        return H

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def add_landmarks(self, number_particles):
        v_add = np.zeros((number_particles, 2))
        v_add_cov = np.zeros((number_particles*2, 2))

        self.lm = np.vstack((self.lm, v_add))
        self.lmP = np.vstack((self.lmP, v_add_cov))

    def marker_array(self, cones_x, cones_y):
        # Publish it as a marker in rviz
        self.marker_ests.markers = []
        for i in range(len(cones_x)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = cones_x[i]
            point.y = cones_y[i]
            point.z = 0.4
            pose.position = point
            orientation = Quaternion()
            # Suponiendo roll y pitch = 0
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            pose.orientation = orientation
            marker_est.pose = pose
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 255, 0)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.2, 0.2, 0.8)
            self.marker_ests.markers.append(marker_est)

    def marker_array_slam(self):
        # Publish it as a marker in rviz
        self.marker_ests.markers = []

        landmarks = self.lm

        # calcular las posiciones medias de los conos en las particulas
        distances_x = landmarks[:, 0]
        distances_y = landmarks[:, 1]

        for i in range(len(distances_x)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = distances_x[i]
            point.y = distances_y[i]
            point.z = 0.4
            pose.position = point
            orientation = Quaternion()
            # Suponiendo roll y pitch = 0
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            pose.orientation = orientation
            marker_est.pose = pose
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 255, 0)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.2, 0.2, 0.8)
            self.marker_ests.markers.append(marker_est)

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
        #print('SLAM', self.yaw)
        #orientation.z = self.yaw
        #orientation.w = 1.0
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
        self.u[1, 0] = yaw

    def update_car_gps_vel(self, velocity):
        self.u[0, 0] = velocity


class Slam_Class(object):
    def __init__(self):
        self.ekf_slam = EKFSLAM()
        self.cones_sub = rospy.Subscriber('/ground_truth/cones', ConeArrayWithCovariance, self.cones_callback)

        self.marker_array_pub = rospy.Publisher('/slam_marker_array_pub', MarkerArray, queue_size=1)

        # Solo pueden estar conectado 1 de estos dos siguientes
        #self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        # self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.state_estimation_callback)

        # Solo pueden estar conectados odom, o imu y gps
        # self.odom_sub = rospy.Subscriber('/odometry_pub', WheelSpeedsStamped, self.odom_callback)
        # self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)

        self.path_pub = rospy.Publisher('/slam_path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/slam_pose_pub', PoseStamped, queue_size=1)

    def sub_callback(self, msg):
        self.ekf_slam.update_car_position(msg.pose.pose)

    def state_estimation_callback(self, msg):
        self.ekf_slam.update_car_position(msg.pose)

    def cones_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        cones = msg.yellow_cones + msg.blue_cones + msg.big_orange_cones
        self.ekf_slam.bucle_principal(cones, timestamp)

    def odom_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_rpm = msg.lb_speed
        velocity_mean = (velocity_rpm * 2.0 * math.pi * 0.25) / 60.0

        # Max steering = 0.52 rad/s
        steering = msg.steering

        self.ekf_slam.update_car_position_odom(timestamp, velocity_mean, steering)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        self.ekf_slam.update_car_gps_vel(velocity_mean)

    def imu_callback(self, msg):
        yaw_rate = msg.angular_velocity.z
        self.ekf_slam.update_car_yaw(yaw_rate)

    def control_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration
        self.ekf_slam.update_car_yaw(angle)


if __name__ == '__main__':
    rospy.init_node('slam_node', anonymous=True)
    slam_class = Slam_Class()
    #rospy.spin()
    rate = rospy.Rate(10)  # Hz

    while not rospy.is_shutdown():
        slam_class.marker_array_pub.publish(slam_class.ekf_slam.marker_ests)
        slam_class.path_pub.publish(slam_class.ekf_slam.path)
        slam_class.pose_pub.publish(slam_class.ekf_slam.pose)
        rate.sleep()
