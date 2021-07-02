#! /usr/bin/env python
import rospy
import numpy as np
import math
from scipy.spatial.distance import cdist
import scipy.interpolate as scipy_interpolate

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from nav_msgs.msg import Path

class Path_planning(object):
    def __init__(self):
        ''' Parametros no configurables '''
        # Estado del coche
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Inicializacion variables path planning
        self.initial_point = np.zeros((1, 2))
        self.state_machine = 0
        self.center_in_view = False

        # Inicializacion de listas para mostrar en Rviz
        self.trayectory_points_x = []
        self.trayectory_points_y = []
        self.marker_ests = MarkerArray()
        self.path = Path()

        ''' Parametros configurables '''
        # Threshold para el filtro
        self.MAX_RANGE = 10

        # Parametros de distancias entre puntos
        self.STARTING_THRESHOLD = 1.5
        self.ORANGE_THRESHOLD = 4.0
        self.DIST_THRESHOLD = 1.0
        self.circle_points = 20
        self.center_distance = 2.0

        # Dimensiones SKIDPAD
        self.trayectory_radius = 9.125
        self.starting_length = 15
        self.finishing_length = 15  # 25?
        self.cone_dist = 3

        # Configuracion del b_spline
        self.n_course_point = 100   # Numero de puntos en la trayectoria final
        self.degree = 3             # Grado de la aproximada B_Spline

    def first_part(self, cones_yellow, cones_blue, cones_orange):
        if len(self.trayectory_points_x) == 0:
            self.trayectory_points_x.append(self.x)
            self.trayectory_points_y.append(self.y)

        midpoint_x_list = []
        midpoint_y_list = []

        cones_x = []
        cones_y = []

        orange_positions = self.array_creation(cones_orange)

        if len(cones_orange) > 0:
            distances = self.calculate_distances(self.x, self.y, orange_positions)
            distances, orange_positions_ordered= map(list, zip(*sorted(zip(distances, cones_orange))))

            # Calculo de puntos medios entre conos naranjas
            for i in range(len(distances) - 1):
                if abs(distances[i] - distances[i+1]) < self.DIST_THRESHOLD:
                    midpoint_x, midpoint_y = self.midpoint(orange_positions_ordered[i].point, orange_positions_ordered[i + 1].point)
                    midpoint_x_list.append(midpoint_x)
                    midpoint_y_list.append(midpoint_y)

        blue_positions = self.array_creation(cones_blue)
        blue_distances = self.calculate_distances(self.x, self.y, blue_positions)
        for i, distance in enumerate(blue_distances):
            if distance < self.MAX_RANGE:
                cont = 0
                distances = self.calculate_distances(cones_blue[i].point.x, cones_blue[i].point.y, orange_positions)
                for dist in distances:
                    if dist <= self.STARTING_THRESHOLD:
                        cont += 1
                if cont == 2:
                    cone_blue = cones_blue[i].point

        yellow_positions = self.array_creation(cones_yellow)
        yellow_positions = self.calculate_distances(self.x, self.y, yellow_positions)
        for i, distance in enumerate(yellow_positions):
            if distance < self.MAX_RANGE:
                cont = 0
                distances = self.calculate_distances(cones_yellow[i].point.x, cones_yellow[i].point.y, orange_positions)
                for dist in distances:
                    if dist <= self.STARTING_THRESHOLD:
                        cont += 1
                if cont == 2:
                    cone_yellow = cones_yellow[i].point

        midpoint_x_list, midpoint_y_list = self.reorganize_trayectory(midpoint_x_list, midpoint_y_list)
        midpoint_x_list, midpoint_y_list = self.filter_points(midpoint_x_list, midpoint_y_list)

        if 'cone_blue' in locals() and 'cone_yellow' in locals():
            midpoint_x, midpoint_y = self.midpoint(cone_blue, cone_yellow)
            midpoint_x_list.append(midpoint_x)
            midpoint_y_list.append(midpoint_y)
            cones_x.append(midpoint_x)
            cones_y.append(midpoint_y)
            self.initial_point[0, 0] = midpoint_x
            self.initial_point[0, 1] = midpoint_y
            self.state_machine = 1

        self.append_new_points(np.array(midpoint_x_list), np.array(midpoint_y_list))

        if len(self.trayectory_points_x) < 2:
            rax = self.trayectory_points_x
            ray = self.trayectory_points_y
        elif len(self.trayectory_points_x) <= self.degree:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, len(self.trayectory_points_x)-1)
        else:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, self.degree)

        self.marker_array_path_planning(cones_x, cones_y)
        self.update_path(rax, ray)

    def second_part(self, cones_yellow, cones_blue, cones_orange):
        # State machine update
        distance = self.calculate_distances(self.x, self.y, self.initial_point)
        if distance < self.center_distance:
            if not self.center_in_view:
                self.center_in_view = True
                self.state_machine += 1
        else:
            self.center_in_view = False

        self.create_trayectory(self.initial_point, self.trayectory_radius)

        if len(self.trayectory_points_x) < 2:
            rax = self.trayectory_points_x
            ray = self.trayectory_points_y
        elif len(self.trayectory_points_x) <= self.degree:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, len(self.trayectory_points_x)-1)
        else:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, self.degree)

        # self.marker_array_path_planning(midpoint_x_list, midpoint_y_list)
        self.update_path(rax, ray)

    def reorganize_trayectory(self, x, y):
        '''
        Elimina los puntos de la trayectoria que esten por delante del coche y los anade a la lista de puntos medios a anadir

        :return: Lista de puntos medios a anadir
        '''
        trayectory_points_x = np.array(self.trayectory_points_x)
        trayectory_points_y = np.array(self.trayectory_points_y)
        trayectory_points_x = trayectory_points_x.reshape(-1, 1)
        trayectory_points_y = trayectory_points_y.reshape(-1, 1)
        trayectory_points = np.hstack((trayectory_points_x, trayectory_points_y))
        distances = self.calculate_distances(self.x, self.y, trayectory_points)
        index_min = np.argmin(distances)

        for i in range(index_min + 1, len(distances)):
            x.append(self.trayectory_points_x.pop(index_min + 1))
            y.append(self.trayectory_points_y.pop(index_min + 1))

        return x, y

    def filter_points(self, x, y):
        '''
        Hace la media de los puntos que se encuentran en cada rango de distancia al coche. Elimina la posicion inicial del
        coche cuando haya superado POINT_NUMBER puntos de trayectoria y cierra la trayectoria cuando vuelve a detectar el
        punto inicial dentro del rango de distancia.

        :param x: Lista de puntos medios en coordenadas globales en el eje x a anadir
        :param y: Lista de puntos medios en coordenadas globales en el eje y a anadir
        :return: Lista de puntos medios de los ejes x e y filtrados
        '''
        x_new = []
        y_new = []

        trayectory_points_x = np.array(x)
        trayectory_points_y = np.array(y)
        trayectory_points_x = trayectory_points_x.reshape(-1, 1)
        trayectory_points_y = trayectory_points_y.reshape(-1, 1)
        trayectory_points = np.hstack((trayectory_points_x, trayectory_points_y))
        distances = self.calculate_distances(self.x, self.y, trayectory_points)

        for i in range(self.MAX_RANGE):
            x_in_range = 0.0
            y_in_range = 0.0
            accumulate = 0.0

            for j, distance in enumerate(distances):
                if distance >= i and distance < (i + 1):
                    x_in_range += trayectory_points_x[j, 0]
                    y_in_range += trayectory_points_y[j, 0]
                    accumulate += 1.0

            if accumulate != 0.0:
                x_new.append(x_in_range / accumulate)
                y_new.append(y_in_range / accumulate)

        return x_new, y_new

    def create_trayectory(self, center, radius):
        if self.state_machine == 2 or self.state_machine == 3:
            point_x = self.trayectory_points_x[-1]
            point_y = self.trayectory_points_y[-1]
            self.trayectory_points_x = []
            self.trayectory_points_y = []
            self.trayectory_points_x.append(point_x)
            self.trayectory_points_y.append(point_y)
            # Primera vuelta
            for i in range(self.circle_points):
                x = center[0, 0] + radius * (np.cos(math.pi / 2 - 2 * math.pi * i / self.circle_points))
                y = center[0, 1] + radius * (-1 + np.sin(math.pi / 2 - 2 * math.pi * i / self.circle_points))
                self.trayectory_points_x.append(x)
                self.trayectory_points_y.append(y)
        elif self.state_machine == 4 or self.state_machine == 5:
            point_x = self.trayectory_points_x[-1]
            point_y = self.trayectory_points_y[-1]
            self.trayectory_points_x = []
            self.trayectory_points_y = []
            self.trayectory_points_x.append(point_x)
            self.trayectory_points_y.append(point_y)
            # Segunda vuelta
            for i in range(self.circle_points):
                x = center[0, 0] + radius * (np.cos(-math.pi / 2 + 2 * math.pi * i / self.circle_points))
                y = center[0, 1] + radius * (1 + np.sin(-math.pi / 2 + 2 * math.pi * i / self.circle_points))
                self.trayectory_points_x.append(x)
                self.trayectory_points_y.append(y)
        elif self.state_machine > 5:
            self.trayectory_points_x = []
            self.trayectory_points_y = []
            # Recta final
            num = 4
            for i in range(num):
                self.trayectory_points_x.append(center[0, 0] + i * self.finishing_length / (num - 1))
                self.trayectory_points_y.append(center[0, 1])

    def array_creation(self, cones):
        '''
        Creacion de un array a partir de los conos.

        :param cones: Posiciones de los conos
        :return: Array de las posiciones de los conos
        '''
        x = np.zeros(len(cones))
        y = np.zeros(len(cones))
        for i, cone in enumerate(cones):
            x[i] = cone.point.x
            y[i] = cone.point.y

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        return np.hstack((x, y))

    def calculate_distances(self, position_x, position_y, points):
        '''
        Calcula las distancias de un punto a una lista de puntos

        :param position_x: Posicion x del punto a comparar
        :param position_y: Posicion y del punto a comparar
        :param points: Lista de puntos a comparar
        :return: Array de distancias del punto a la lista de puntos
        '''
        position = np.array([(position_x, position_y)])
        distances = cdist(position, points, 'euclidean')
        distances = distances.reshape(-1, )

        return distances

    def midpoint(self, yellow, blue):
        '''
        Funcion que devuelve el punto medio entre las coordenadas de un cono amarillo y otro azul.

        :param yellow: Coordenadas locales del cono amarillo
        :param blue: Coordenadas locales del cono azul
        :return: Coordenadas globales del punto medio
        '''
        midpoint_x = (yellow.x + blue.x) / 2.0
        midpoint_y = (yellow.y + blue.y) / 2.0
        midpoint_x, midpoint_y = self.local_to_global(midpoint_x, midpoint_y)
        return midpoint_x, midpoint_y

    def approximate_b_spline_path(self, x, y, n_path_points, degree):
        """
        Aproximar los puntos mediante una B-Spline.

        :param x: Lista de puntos que debe seguir en el eje x
        :param y: Lista de puntos que debe seguir en el eje y
        :param n_path_points: Numero de puntos totales para la trayectoria
        :param degree: (Opcional) Grado de la curva B Spline
        :return: Listas de los puntos en el eje x e y de la trayectoria final
        """
        t = range(len(x))
        x_tup = scipy_interpolate.splrep(t, x, k=degree)
        y_tup = scipy_interpolate.splrep(t, y, k=degree)

        x_list = list(x_tup)
        x_list[1] = x

        y_list = list(y_tup)
        y_list[1] = y

        ipl_t = np.linspace(0.0, len(x) - 1, n_path_points)
        rx = scipy_interpolate.splev(ipl_t, x_list)
        ry = scipy_interpolate.splev(ipl_t, y_list)

        return rx, ry

    def append_new_points(self, x, y):
        '''
        Anade a la lista de puntos de la trayectoria los nuevos puntos medios, ordenados segun la distancia al ultimo
        punto anadido

        :param x: Lista de puntos medios en coordenadas globales en el eje x a anadir
        :param y: Lista de puntos medios en coordenadas globales en el eje y a anadir
        '''
        loop = len(x)
        for i in range(loop):
            rx = x.reshape(-1, 1)
            ry = y.reshape(-1, 1)
            cone_positions = np.hstack((rx, ry))
            distances = self.calculate_distances(self.trayectory_points_x[-1], self.trayectory_points_y[-1], cone_positions)
            index_min = np.argmin(distances)
            self.trayectory_points_x.append(x[index_min])
            self.trayectory_points_y.append(y[index_min])
            x = np.delete(x, index_min)
            y = np.delete(y, index_min)

    def local_to_global(self, x, y):
        '''
        Pasa de coordenadas locales a globales segun la posicion del coche obtenida del state estimation.

        :param x: Valor en el eje x en coordenadas locales
        :param y: Valor en el eje y en coordenadas locales
        :return: Coordenadas globales del punto
        '''
        position = np.array([x, y])
        C_ns = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
        rotated_position = C_ns.dot(position) + np.array([self.x, self.y])

        return rotated_position[0], rotated_position[1]

    def pi_2_pi(self, angle):
        '''
        Devuelve el angulo en un rango entre -pi y pi.

        :param angle: Angulo de entrada
        :return: Angulo pasado al rango [-pi, pi]
        '''
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update_car_position(self, pose):
        '''
        Actualizacion de la posicion del coche.

        :param pose: Estado actual del coche
        '''
        self.x = pose.position.x
        self.y = pose.position.y
        self.yaw = self.pi_2_pi(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))

    def update_path(self, x, y):
        '''
        Actualizacion de la trayectoria que se publica para control y Rviz.

        :param x: Lista de puntos globales en el eje x de la trayectoria
        :param y: Lista de puntos globales en el eje y de la trayectoria
        '''
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        for i in range(len(x)):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            point = Point()
            point.x = x[i]
            point.y = y[i]
            point.z = 0.0
            pose.pose.position = point
            orientation = Quaternion()
            # Suponiendo roll y pitch = 0
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            pose.pose.orientation = orientation
            path.poses.append(pose)

        self.path = path

    def marker_array_path_planning(self, midpoint_x_list, midpoint_y_list):
        '''
        MarkerArray para mostrar en Rviz

        :param midpoint_x_list: Puntos en el eje x a mostrar
        :param midpoint_y_list: Puntos en el eje y a mostrar
        '''
        # Publish it as a marker in rviz
        self.marker_ests.markers = []

        for i in range(len(midpoint_x_list)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = midpoint_x_list[i]
            point.y = midpoint_y_list[i]
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
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 255, 0)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.2, 0.2, 0.8)
            self.marker_ests.markers.append(marker_est)