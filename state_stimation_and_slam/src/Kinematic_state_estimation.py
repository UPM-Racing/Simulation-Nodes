#! /usr/bin/env python
import rospy
import numpy as np
import math
import copy
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3Stamped
from GPS import GPS

from eufs_msgs.msg import WheelSpeedsStamped
from visualization_msgs.msg import MarkerArray, Marker

class Car():
    def __init__(self):
        ''' Variables no configurables '''
        # Estado del coche
        self.x = np.zeros(2)
        self.v = 0.0
        self.a = 0.0
        self.yaw = 0.0
        self.p_cov = np.eye(5)

        # Inicializacion de variables
        self.past_time = 0.0
        self.imu_past_time = 0.0
        self.odom_past_time = 0.0
        self.car_created = False
        self.h_jac = np.zeros([2, 5])       # Modelo del jacobiano de posicion
        self.h_jac[:, :2] = np.eye(2)
        self.h_vel_jac = np.zeros([1, 5])   # Modelo del jacobiano de velocidad
        self.h_vel_jac[:, 2] = 1.0
        self.h_steer_jac = np.zeros([1, 5]) # Modelo del jacobiano de giro
        self.h_steer_jac[:, 4] = np.eye(1)

        # Inicializacion de variables para Rviz
        self.path = Path()
        

        self.pose = PoseStamped()
        self.control_msg = WheelSpeedsStamped()
        self.marker_ests = MarkerArray()

        ''' Variables configurables '''
        # Datos del coche
        self.radio = 0.2525     # Radio de la rueda del coche
        self.wheelbase = 1.58   # Distancia entre ejes del coche
        self.lr = 0.711         # Distancia del centro de gravedad al eje trasero del coche

        # Varianzas de los sensores
        self.var_control_a = 0.10 * 10
        self.var_control_w = 0.10 * 10
        self.var_gnss = 0.10 * 10
        self.var_odom = 0.10
        self.var_imu = 0.10 * 10

    def InicializacionCoche(self, time):
        '''
        Inicializacion del coche.

        :param time: Instante de tiempo actual
        '''
        self.past_time = time
        self.car_created = True

    def Kinematic_prediction(self, time, u):
        '''
        Modelo cinematico del coche basado en el bicycle model. Entra con cada nuevo mensaje de control y actualiza la
        posicion del coche.

        :param time: Instante de tiempo actual
        :param u: Entradas del modelo: aceleracion y giro del volante (dados por control)
        '''
        # Actualiza la medida de tiempo
        delta_t = time - self.past_time
        self.past_time = time

        beta = math.atan2(self.lr * math.tan(u[1]), self.wheelbase)

        x_dot = self.v * math.cos(self.yaw + beta)
        y_dot = self.v * math.sin(self.yaw + beta)
        yaw_dot = self.v * math.sin(beta) / self.lr

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
                      [0, 0, (math.sin(beta) / self.lr) * delta_t, 0, 1]])

        Q = np.eye(5)
        Q[0:2, 0:2] = self.var_control_a * Q[0:2, 0:2]
        Q[2:3, 2:3] = self.var_control_a * Q[2:3, 2:3]
        Q[3:4, 3:4] = self.var_control_a * Q[3:4, 3:4]
        Q[4:5, 4:5] = self.var_control_w * Q[4:5, 4:5]
        Q = (delta_t ** 2) * Q

        # Propagar la incertidumbre
        self.p_cov = F.dot(self.p_cov).dot(F.T) + Q

        self.update_path()

    def pi_2_pi(self, angle):
        '''
        Devuelve el angulo en un rango entre -pi y pi.

        :param angle: Angulo de entrada
        :return: Angulo pasado al rango [-pi, pi]
        '''
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def updateStep(self, gps_coord):
        '''
        Actualiza la posicion del coche a partir de la posicion proveniente del gps.

        :param gps_coord: Posicion del coche dada por el gps en coordenadas locales.
        '''
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
        '''
        Actualiza la velocidad del coche a partir del valor de velocidad entrante.

        :param velocity: Velocidad
        '''
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

    def updateStepAcc(self, time, acceleration):
        '''
        Actualiza la velocidad del coche a partir del valor de aceleracion entrante.

        :param velocity: Velocidad
        '''
        delta_t = time - self.past_time
        h_vel_jac = self.h_vel_jac * delta_t

        # 3.1 Compute Kalman Gain
        R = self.var_imu * np.eye(1)
        K = self.p_cov.dot(h_vel_jac.T.dot(np.linalg.inv(h_vel_jac.dot(self.p_cov.dot(h_vel_jac.T)) + R)))

        # 3.2 Compute error state
        delta_v = K.dot(np.array([delta_t * acceleration]))

        # 3.3 Correct predicted state
        self.x = self.x + delta_v[:2]
        self.v = self.v + delta_v[2]
        self.a = self.a + delta_v[3]
        self.yaw = self.yaw + delta_v[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(h_vel_jac)).dot(self.p_cov)

    def updateStepSteer(self, time, steering):
        delta_t = time - self.past_time
        h_steer_jac = self.h_steer_jac * delta_t

        beta = math.atan2(self.lr * math.tan(steering), self.wheelbase)
        yaw_dot = self.v * math.sin(beta) / self.lr

        # 3.1 Compute Kalman Gain
        R = self.var_imu * np.eye(1)
        K = self.p_cov.dot(h_steer_jac.T.dot(np.linalg.inv(h_steer_jac.dot(self.p_cov.dot(h_steer_jac.T)) + R)))

        # 3.2 Compute error state
        delta_yaw = K.dot(np.array([yaw_dot * delta_t]))

        # 3.3 Correct predicted state
        self.x = self.x + delta_yaw[:2]
        self.v = self.v + delta_yaw[2]
        self.a = self.a + delta_yaw[3]
        self.yaw = self.yaw + delta_yaw[4]

        # 3.4 Compute corrected covariance
        self.p_cov = (np.eye(5) - K.dot(h_steer_jac)).dot(self.p_cov)

    def update_path(self):
        '''
        Actualizacion de la trayectoria de la posicion estimada del coche para Rviz.

        :param x: Lista de puntos globales en el eje x de la trayectoria
        :param y: Lista de puntos globales en el eje y de la trayectoria
        '''
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
        self.plot_covariance_ellipse()  

    def publish_control(self):
        '''
        Actualizar la velocidad en el mensaje de control.

        :return: Mensaje de control actualizado con la velocidad
        '''
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.header.frame_id = "map"
        self.control_msg.rf_speed = self.v

        return self.control_msg

    def plot_covariance_ellipse(self):
        '''
        Saca los autovalores y autovectores de la matriz de covarianza de la posicion para dibujar una elipse.

        '''
        Pxy = self.p_cov[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)
        angle = math.acos(eigvec[0, 0])

        self.marker_array(eigval, angle)

    def marker_array(self, eigval, angle):
        '''
        Actualiza la variable del MarkerArray de la elipse de la covarianza para publicarlo en Rviz.

        :param eigval: Autovalores de la matriz de covarianza de la posicion
        :param angle: Angulo del autovector
        '''
        # Publish it as a marker in rviz
        self.marker_ests.markers = []
        marker_est = Marker()
        marker_est.header.frame_id = "map"
        marker_est.ns = "est_pose"
        marker_est.id = 0
        marker_est.type = Marker.CYLINDER
        marker_est.action = Marker.ADD
        pose = Pose()
        point = Point()
        point.x = self.x[0]
        point.y = self.x[1]
        point.z = 0.255
        pose.position = point
        orientation = Quaternion()
        # Suponiendo roll y pitch = 0
        orientation.x = 0.0
        orientation.y = 0.0
        orientation.z = np.sin(angle * 0.5)
        orientation.w = np.cos(angle * 0.5)
        pose.orientation = orientation
        marker_est.pose = pose
        marker_est.color.r, marker_est.color.g, marker_est.color.b = (255, 0, 0)
        marker_est.color.a = 0.5
        marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (eigval[0], eigval[1], 0.01)
        self.marker_ests.markers.append(marker_est)

