#! /usr/bin/env python

import rospy
import numpy as np
import math
from eufs_msgs.msg import ConeArrayWithCovariance, CarState
from scipy.spatial.distance import cdist
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
import scipy.interpolate as scipy_interpolate
from nav_msgs.msg import Path

class Path_planning(object):
    def __init__(self):
        ''' Parametros no configurables '''
        # Estado del coche
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Inicializacion variables path planning
        self.finished_lap = False
        self.deleted_initial_point = False

        # Inicializacion de listas para mostrar en Rviz
        self.trayectory_points_x = []
        self.trayectory_points_y = []
        self.marker_ests = MarkerArray()
        self.path = Path()

        ''' Parametros configurables '''
        # Threshold de distancia entre conos
        self.THRESHOLD = 5.5
        self.ORANGE_THRESHOLD = 2.5
        self.CLEAR_THRESHOLD = 1.0
        self.REORGANIZE_THRESHOLD = 5.0
        self.POINT_NUMBER = 20.0

        # Configuracion del b_spline
        self.n_course_point = 100   # sampling number
        self.degree = 3             # approximate_b_spline_path degree

    def bucle_principal(self, cones_yellow, cones_blue, cones_orange):
        '''
        Este es el bucle principal de path planning, que se ejecuta cada vez que llega un mensaje de observaciones.

        :param cones_yellow: Coordenadas locales de la posicion de los conos amarillos
        :param cones_blue: Coordenadas locales de la posicion de los conos azules
        :param cones_orange: Coordenadas locales de la posicion de los conos naranjas
        :return:
        '''
        cones_for_array_x = []
        cones_for_array_y = []
        midpoint_x_list = []
        midpoint_y_list = []
        blue_x = np.zeros(len(cones_blue))
        blue_y = np.zeros(len(cones_blue))
        for i, cone in enumerate(cones_blue):
            blue_x[i] = cone.point.x
            blue_y[i] = cone.point.y
            cone_for_array_x, cone_for_array_y = self.local_to_global(cone.point.x, cone.point.y)
            cones_for_array_x.append(cone_for_array_x)
            cones_for_array_y.append(cone_for_array_y)

        for i, cone in enumerate(cones_yellow):
            cone_for_array_x, cone_for_array_y = self.local_to_global(cone.point.x, cone.point.y)
            cones_for_array_x.append(cone_for_array_x)
            cones_for_array_y.append(cone_for_array_y)

        blue_x = blue_x.reshape(-1, 1)
        blue_y = blue_y.reshape(-1, 1)
        blue_positions = np.hstack((blue_x, blue_y))

        for i, cone in enumerate(cones_yellow):
            cone_x = cone.point.x
            cone_y = cone.point.y
            cone_position = np.array([(cone_x, cone_y)])
            distances = cdist(cone_position, blue_positions, 'euclidean')
            distances = distances.reshape(-1,)
            for j, distance in enumerate(distances):
                if distance <= self.THRESHOLD:
                    midpoint_x, midpoint_y = self.midpoint(cone.point, cones_blue[j].point)
                    midpoint_x_list.append(midpoint_x)
                    midpoint_y_list.append(midpoint_y)

        orange_x = np.zeros(len(cones_orange))
        orange_y = np.zeros(len(cones_orange))
        for i, cone in enumerate(cones_orange):
            orange_x[i] = cone.point.x
            orange_y[i] = cone.point.y
            cone_for_array_x, cone_for_array_y = self.local_to_global(cone.point.x, cone.point.y)
            cones_for_array_x.append(cone_for_array_x)
            cones_for_array_y.append(cone_for_array_y)

        orange_x = orange_x.reshape(-1, 1)
        orange_y = orange_y.reshape(-1, 1)
        orange_positions = np.hstack((orange_x, orange_y))

        if len(cones_orange) > 0:
            cone_x = cones_orange[0].point.x
            cone_y = cones_orange[0].point.y
            orange_positions = np.delete(orange_positions, 0, 0)
            cone_position = np.array([(cone_x, cone_y)])
            distances = cdist(cone_position, orange_positions, 'euclidean')
            distances = distances.reshape(-1,)
            for j, distance in enumerate(distances):
                if distance >= self.ORANGE_THRESHOLD:
                    midpoint_x, midpoint_y = self.midpoint(cones_orange[0].point, cones_orange[j+1].point)
                    midpoint_x_list.append(midpoint_x)
                    midpoint_y_list.append(midpoint_y)

        if len(self.trayectory_points_x) == 0 and len(midpoint_x_list) > 0:
            self.trayectory_points_x.append(self.x)
            self.trayectory_points_y.append(self.y)
            self.append_new_points(np.array(midpoint_x_list), np.array(midpoint_y_list))
        elif len(midpoint_x_list) > 0:
            x_new, y_new = self.clear_points(midpoint_x_list, midpoint_y_list)
            self.append_new_points(np.array(x_new), np.array(y_new))
            self.reorganize_trayectory()

        if len(self.trayectory_points_x) < 2:
            rax = self.trayectory_points_x
            ray = self.trayectory_points_y
        elif len(self.trayectory_points_x) <= self.degree:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, len(self.trayectory_points_x)-1)
        else:
            rax, ray = self.approximate_b_spline_path(self.trayectory_points_x, self.trayectory_points_y, self.n_course_point, self.degree)

        self.marker_array_path_planning(midpoint_x_list, midpoint_y_list)
        self.update_path(rax, ray)

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
            last_position = np.array([(self.trayectory_points_x[-1], self.trayectory_points_y[-1])])
            rx = x.reshape(-1, 1)
            ry = y.reshape(-1, 1)
            cone_positions = np.hstack((rx, ry))
            distances = cdist(last_position, cone_positions, 'euclidean')
            distances = distances.reshape(-1, )
            index_min = np.argmin(distances)
            self.trayectory_points_x.append(x[index_min])
            self.trayectory_points_y.append(y[index_min])
            x = np.delete(x, index_min)
            y = np.delete(y, index_min)

    def clear_points(self, x, y):
        '''
        Elimina los puntos medios que esten mas cerca entre si que el CLEAR_THRESHOLD. Elimina la posicion inicial del
        coche cuando haya superado POINT_NUMBER puntos de trayectoria.

        :param x: Lista de puntos medios en coordenadas globales en el eje x a anadir
        :param y: Lista de puntos medios en coordenadas globales en el eje y a anadir
        :return: Lista de puntos medios de los ejes x e y filtrados
        '''
        x_new = []
        y_new = []

        trayectory_points_x = np.array(self.trayectory_points_x)
        trayectory_points_y = np.array(self.trayectory_points_y)
        trayectory_points_x = trayectory_points_x.reshape(-1, 1)
        trayectory_points_y = trayectory_points_y.reshape(-1, 1)

        for i in range(len(x)):
            trayectory_points = np.hstack((trayectory_points_x, trayectory_points_y))
            position = np.array([(x[i], y[i])])
            distances = cdist(position, trayectory_points, 'euclidean')
            distances = distances.reshape(-1, )
            index_min = np.argmin(distances)

            if len(self.trayectory_points_x) > self.POINT_NUMBER and self.deleted_initial_point is False:
                self.trayectory_points_x.pop(0)
                self.trayectory_points_y.pop(0)
                self.deleted_initial_point = True

            if distances[0] <= self.CLEAR_THRESHOLD and len(self.trayectory_points_x) > self.POINT_NUMBER:
                self.finished_lap = True
                x_new.append(x[i])
                y_new.append(y[i])
                # print('FINISHED LAP')
            elif distances[index_min] >= self.CLEAR_THRESHOLD:
                x_new.append(x[i])
                y_new.append(y[i])
                np.append(trayectory_points_x, x[i])
                np.append(trayectory_points_y, y[i])

        return x_new, y_new

    def reorganize_trayectory(self):
        '''
        Elimina los puntos de la trayectoria que esten a una distancia REORGANIZE_THRESHOLD del coche

        :return:
        '''
        car_position = np.array([(self.x, self.y)])
        trayectory_points_x = np.array(self.trayectory_points_x)
        trayectory_points_y = np.array(self.trayectory_points_y)
        trayectory_points_x = trayectory_points_x.reshape(-1, 1)
        trayectory_points_y = trayectory_points_y.reshape(-1, 1)
        trayectory_points = np.hstack((trayectory_points_x, trayectory_points_y))
        distances = cdist(car_position, trayectory_points, 'euclidean')
        distances = distances.reshape(-1, )

        x_new = []
        y_new = []
        index = 0

        for i in range(1, len(distances)):
            if distances[i] <= self.REORGANIZE_THRESHOLD:
                index = i
                break

        if index != 0:
            for i in range(index, len(distances)):
                x_new.append(self.trayectory_points_x.pop(index))
                y_new.append(self.trayectory_points_y.pop(index))

        self.append_new_points(np.array(x_new), np.array(y_new))

    def local_to_global(self, x, y):
        position = np.array([x, y])
        C_ns = np.array([[np.cos(self.yaw), -np.sin(self.yaw)], [np.sin(self.yaw), np.cos(self.yaw)]])
        rotated_position = C_ns.dot(position) + np.array([self.x, self.y])

        #d, angle = self.definir_posicion(x, y)

        # calculamos el seno y el coseno del angulo del coche mas el de la observacion
        #s = math.sin(self.pi_2_pi(self.yaw + self.pi_2_pi(angle)))
        #c = math.cos(self.pi_2_pi(self.yaw + self.pi_2_pi(angle)))

        # se anaden la x y la y correspondientes a la lista de landmarks
        #x = self.x + d * c
        #y = self.y + d * s

        return rotated_position[0], rotated_position[1]

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update_car_position(self, pose):
        self.x = pose.position.x
        self.y = pose.position.y
        #print('----------------')
        #print('Path planning', self.yaw)
        self.yaw = self.pi_2_pi(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))
        #print('Path planning', self.yaw)
        #self.yaw = pose.orientation.z

    def update_path(self, x, y):
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

class Path_planning_class(object):
    def __init__(self):
        self.path_planning = Path_planning()
        self.cones_sub = rospy.Subscriber('/ground_truth/cones', ConeArrayWithCovariance, self.cones_callback)

        self.marker_array_pub = rospy.Publisher('/path_planning_marker_array_pub', MarkerArray, queue_size=1)

        #self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.sub_callback2)
        #self.slam_pose_sub = rospy.Subscriber('/slam_pose_pub', PoseStamped, self.sub_callback2)

        self.path_pub = rospy.Publisher('/path_planning_pub', Path, queue_size=1)

    def sub_callback(self, msg):
        self.path_planning.update_car_position(msg.pose.pose)
        #print('------GROUND TRUTH---------')
        #print(self.path_planning.pi_2_pi(np.arctan2(2 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z), 1 - 2 * (msg.pose.pose.orientation.z ** 2))))

    def sub_callback2(self, msg):
        self.path_planning.update_car_position(msg.pose)

    def cones_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        cones_yellow = msg.yellow_cones
        cones_blue = msg.blue_cones
        cones_orange = msg.big_orange_cones
        if not self.path_planning.finished_lap:
            self.path_planning.bucle_principal(cones_yellow, cones_blue, cones_orange)


if __name__ == '__main__':
    rospy.init_node('path_planning_node', anonymous=True)
    path_class = Path_planning_class()
    #rospy.spin()
    rate = rospy.Rate(10)  # Hz

    while not rospy.is_shutdown():
        '''if len(path_class.path_planning.trayectory_points_x) > 3:
            rax, ray = path_class.path_planning.approximate_b_spline_path(path_class.path_planning.trayectory_points_x,
                                                                          path_class.path_planning.trayectory_points_y,
                                                                          path_class.path_planning.n_course_point,
                                                                          path_class.path_planning.degree)
            #path_class.path_planning.marker_array_path_planning(rax, ray)
            path_class.path_planning.update_path(rax, ray)'''

        path_class.marker_array_pub.publish(path_class.path_planning.marker_ests)
        path_class.path_pub.publish(path_class.path_planning.path)
        rate.sleep()