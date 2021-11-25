#! /usr/bin/env python
import rospy
import math
import numpy as np
import time
import csv

from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker


class SLAM:
    def __init__(self):
        ''' Variables no configurables '''
        # Estado del coche
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.xEst = np.zeros((3, 1))
        self.PEst = np.eye(3)

        # Inicializacion de variables
        self.past_time = 0.0
        self.DT = 0.0
        self.u = np.zeros((2, 1))
        self.path = Path()
        self.pose = PoseStamped()
        self.slam_marker_ests = MarkerArray()
        self.h_jac = np.zeros([2, 3])
        self.h_jac[:, :2] = np.eye(2)  # Jacobiano del modelo cinematico
        self.cones_to_csv = False

        # Variables no usadas en principio
        self.state_x = 0.0
        self.state_y = 0.0
        self.state_yaw = 0.0

        ''' Variables configurables '''
        # Datos del coche
        self.radio = 0.2525  # Radio de la rueda del coche
        self.wheelbase = 1.58  # Distancia entre ejes del coche
        self.lr = 0.711  # Distancia del centro de gravedad al eje trasero del coche

        # Constantes para EKF SLAM
        self.Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
        self.THRESHOLD = 1.0
        self.M_DIST_TH = 0.3  # Threshold of Mahalanobis distance for data association.
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]

    def bucle_principal(self, cones, timestamp):
        '''
        Bucle principal de actualizacion del estado del coche. Se llama cada vez que llega un mensaje de conos. Crea el
        vector de distancias y angulos locales a las observaciones y llama a la funcion de slam.

        :param cones: Lista de la posicion de todos los conos observables
        :param time: Instante de tiempo actual
        '''
        '''period = time.time()'''
        self.DT = timestamp - self.past_time
        self.past_time = timestamp

        z = np.zeros((0, 2))

        for i, cone in enumerate(cones):
            d, angle = self.definir_posicion(cone.point.x, cone.point.y)
            zi = np.array([d, angle])
            z = np.vstack((z, zi))

        self.xEst, self.PEst = self.ekf_slam(self.xEst, self.PEst, self.u, z)

        self.x = self.xEst[0]
        self.y = self.xEst[1]
        self.yaw = self.xEst[2]

        '''period = time.time() - period
        with open('../catkin_ws/results/Ekf_Slam.csv', 'ab') as csvfile:
            writer=csv.writer(csvfile, delimiter='\t',lineterminator='\n',)
            writer.writerow([period])'''

        # Actualiza las variables de Rviz
        self.rviz_update()

        '''if self.calc_n_lm(self.xEst) == 69 and not self.cones_to_csv:
            with open('../catkin_ws/results/cones_position_ekf_slam.csv', 'ab') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
                for i in range(self.calc_n_lm(self.xEst)):
                    lm = self.get_landmark_position_from_state(self.xEst, i)
                    writer.writerow([i, lm[0], lm[1]])
                print('done')
                self.cones_to_csv = True'''

    def ekf_slam(self, xEst, PEst, u, z):
        '''
        Funcion global de EKF SLAM. Calcula el nuevo estado del coche, y lo ajusta segun las nuevas observaciones,
        comprobando si son nuevos landmarks o landmarks ya guardados.

        :param xEst: Vector de estado del coche y landmarks
        :param PEst: Matriz de covarianzas del coche y landmarks
        :param u: Vector de entrada al modelo [velocidad, giro de volante]
        :param z: Vector con las distancias y angulos locales con las que el coche ve las observaciones
        :return: Vector de estado del coche y landmarks y matriz de covarianzas del coche y landmarks actualizados
        '''
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
            lm = self.get_landmark_position_from_state(xEst, min_id)
            y, S, H = self.calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

            K = np.matmul(np.matmul(PEst, H.T), np.linalg.inv(S))
            xEst = xEst + np.matmul(K, y)
            PEst = np.matmul(np.eye(len(xEst)) - np.matmul(K, H), PEst)

        xEst[2] = self.pi_2_pi(xEst[2])

        return xEst, PEst

    def definir_posicion(self, x, y):
        '''
        Funcion que devuelve la distancia y angulo en coordenadas locales de un punto.

        :param x: Coordenada x en locales de un punto
        :param y: Coordenada y en locales de un punto
        :return: Distancia y angulo al punto
        '''
        d = math.hypot(x, y)
        angle = self.pi_2_pi(math.atan2(y, x))
        return d, angle

    def motion_model(self, x, u):
        '''
        Modelo cinematico basado en el kinematic bicycle model del coche.

        :param x: Vector de estado del coche anterior
        :param u: Vector de entrada al modelo [velocidad, giro de volante]
        :return: Vector de estado del coche y landmarks con el estado del coche actualizado
        '''
        beta = math.atan2(self.lr * math.tan(u[1, 0]), self.wheelbase)

        x_dot = u[0, 0] * math.cos(x[2, 0] + beta)
        y_dot = u[0, 0] * math.sin(x[2, 0] + beta)
        yaw_dot = u[0, 0] * math.sin(beta) / self.lr

        x[0, 0] = x[0, 0] + x_dot * self.DT
        x[1, 0] = x[1, 0] + y_dot * self.DT
        x[2, 0] = x[2, 0] + yaw_dot * self.DT

        x[2, 0] = self.pi_2_pi(x[2, 0])

        return x

    def calc_n_lm(self, x):
        '''
        Calcula el numero de landmarks almacenados en x.

        :param x: Vector de estado del coche y landmarks
        :return: Numero de landmarks almacenados en x
        '''
        n = int((len(x) - self.STATE_SIZE) / self.LM_SIZE)
        return n

    def jacob_motion(self, x, u):
        '''
        Calcula los jacobianos del modelo del coche

        :param x: Vector de estado del coche
        :param u: Vector de entrada al modelo [velocidad, giro de volante]
        :return:
        '''
        Fx = np.hstack((np.eye(self.STATE_SIZE), np.zeros((self.STATE_SIZE, self.LM_SIZE * self.calc_n_lm(x)))))

        jF = np.array([[0.0, 0.0, -self.DT * u[0, 0] * math.sin(x[2, 0])],
                       [0.0, 0.0, self.DT * u[0, 0] * math.cos(x[2, 0])],
                       [0.0, 0.0, 0.0]], dtype=float)

        G = np.eye(self.STATE_SIZE) + np.matmul(np.matmul(Fx.T, jF), Fx)

        return G, Fx

    def calc_landmark_position(self, x, z):
        '''
        Calcula la posicion en ejes globales de un landmark.

        :param x: Vector de estado del coche y landmarks en ejes globales
        :param z: Distancia y angulo del coche al landmark en ejes locales
        :return: Posicion en ejes globales del landmark
        '''
        zp = np.zeros((2, 1))

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

        return zp

    def get_landmark_position_from_state(self, x, ind):
        '''
        Devuelve la posicion del landmark indicado.

        :param x: Vector de estado del coche y landmarks
        :param ind: Indice del landmark deseado
        :return: Posicion del landmark deseado
        '''
        lm = x[self.STATE_SIZE + self.LM_SIZE * ind: self.STATE_SIZE + self.LM_SIZE * (ind + 1), :]

        return lm

    def search_correspond_landmark_id(self, xEst, PEst, zi):
        """
        Comprueba si la nueva observacion es alguna de las landmarks guardadas o es una nueva, calculando la distancia
        de Mahalanobis para cada landmark y comparandolas con un threshold de distancia.

        :param xEst: Vector de estado del coche y landmarks
        :param PEst: Matriz de covarianzas del vector de estado y posicion de los landmarks
        :param zi: Posicion local de una observacion (d, angle)
        :return: Id del landmark que cumpla un threshold de distancia o anade un id nuevo
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
        '''
        Calcula la diferencia entre la posicion guardada del landmark y la posicion del landmark dada por percepcion.

        :param lm: Posicion del landmark guardado
        :param xEst: Vector de estado del coche y landmarks
        :param PEst: Matriz de covarianzas del vector de estado y posicion de los landmarks
        :param z: Posicion local de una observacion (d, angle)
        :param LMid: Id del landmark observado
        :return:
            y: Diferencia de distancia y angulo entre la posicion guardada y la nueva observacion
            S: Matriz de covarianza de la posicion del landmark
            H: Jacobiano de la posicion del landmark
        '''
        delta = lm - xEst[0:2]
        q = (np.matmul(delta.T, delta))[0, 0]
        z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
        zp = np.array([[math.sqrt(q), self.pi_2_pi(z_angle)]])
        y = (z - zp).T
        y[1] = self.pi_2_pi(y[1])
        H = self.jacob_h(q, delta, xEst, LMid)
        S = np.matmul(np.matmul(H, PEst), H.T) + self.Cx[0:2, 0:2]

        return y, S, H

    def jacob_h(self, q, delta, x, i):
        '''
        Calcula el jacobiano de la posicion del landmark.

        :param q: Distancia del coche al landmark guardados al cuadrado
        :param delta: Diferencia de posicion en ambos ejes del coche al landmark guardados
        :param x: Vector de estado del coche y landmarks
        :param i: Id del landmark
        :return: Jacobiano de la posicion del landmark
        '''
        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                      [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

        G = G / q
        nLM = self.calc_n_lm(x)
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * i)),
                        np.eye(2), np.zeros((2, 2 * nLM - 2 * (i + 1)))))

        F = np.vstack((F1, F2))

        H = np.matmul(G, F)

        return H

    def pi_2_pi(self, angle):
        '''
        Devuelve el angulo en un rango entre -pi y pi.

        :param angle: Angulo de entrada
        :return: Angulo pasado al rango [-pi, pi]
        '''
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def rviz_update(self):
        '''
        Actualiza las variables que se mostraran en Rviz

        '''
        cones_x = []
        cones_y = []

        # Landmarks
        for i in range(self.calc_n_lm(self.xEst)):
            cones_x.append(self.xEst[3 + i * 2])
            cones_y.append(self.xEst[3 + i * 2 + 1])

        self.marker_array(cones_x, cones_y)
        self.publish_path()

    def marker_array(self, cones_x, cones_y):
        '''
        MarkerArray para actualizar y mostrar en Rviz la posicion de los conos que estan guardados.

        :param cones_x: Lista con la posicion en el eje x de los conos
        :param cones_y: Lista con la posicion en el eje y de los conos
        '''
        # Publish it as a marker in rviz
        self.slam_marker_ests.markers = []
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
            self.slam_marker_ests.markers.append(marker_est)

    def publish_path(self):
        '''
        Actualizacion de la trayectoria del coche que se publica para control y Rviz.

        '''
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

    def update_car_steer(self, steer):
        '''
        Actualiza la entrada de giro de volante dada por control

        :param steer: Giro de volante del coche dado por control
        '''
        self.u[1, 0] = steer

    def update_car_gps_vel(self, velocity):
        '''
        Actualiza la entrada de velocidad dada por gps_velocity

        :param velocity: Velocidad dada por gps_velocity
        '''
        self.u[0, 0] = velocity

