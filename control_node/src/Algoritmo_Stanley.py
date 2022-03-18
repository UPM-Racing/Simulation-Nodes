#! /usr/bin/env python

from cmath import sqrt
import numpy as np
import math
import scipy.interpolate as scipy_interpolate


Kp = 1.0                                    # speed proportional gain
ki = 1.0                                    # speed integral gain
kd = 0.1                                    # speed derivational gain
dt = 0.1                                    # [s] time difference
target_speed = 10.0 / 3.6                   # [m/s]
Ke = 5                                      # control gain
Kv = 1
max_steer = 27.2 * np.pi / 180              # [rad] max steering angle
max_accel = 1.0                             # [m/s^2] max acceleration
indice_prev=0 
vuelta_reconomiento=0

VEL_THRESHOLD = 1e-4





class Stanley(object):
    def __init__(self):
        self.v = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.n_course_point = 200
        self.degree = 3

        self.steering_angle = 0.0
        self.acceleration = 0.0

        self.flag_entrada_circulo=0

        self.contador_de_vuelta=0.0
        
        self.rmse_acumulado = 0.0
        self.error = []
        self.error_cuadrado = []
        self.error_maximo = 0.0
        self.finish_flag=0 

    def principal_loop(self, path_planning):
        poses = path_planning.poses
        positions_x = []
        positions_y = []
        global vuelta_reconomiento



        for i, pose in enumerate(poses):
            positions_x.append(pose.pose.position.x)
            positions_y.append(pose.pose.position.y)
        
        path_x, path_y = self.interpolate_b_spline_path(positions_x, positions_y, self.n_course_point, self.degree)

        if len(positions_x) > self.degree:
            path_x, path_y = self.interpolate_b_spline_path(positions_x, positions_y, self.n_course_point, self.degree)
        elif len(positions_x) >= 2:
            path_x, path_y = self.interpolate_b_spline_path(positions_x, positions_y, self.n_course_point, len(positions_x) - 1)

        if 'path_x' in locals():
            self.stanley_control(path_x, path_y)
            self.pid_control(target_speed)

        if self.finish_flag==1 and self.v<VEL_THRESHOLD     :
            print("mision status =finish ")

        if (positions_x[0]==positions_x[-1] and  positions_y[0]==positions_y[-1]) and vuelta_reconomiento==0:
            print ("vuelta de reconocimiento ")
            vuelta_reconomiento=1

       


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
        self.acceleration = np.clip(accel, -max_accel, max_accel)        

        #print("Aceleracion: {}".format(accel))

    def stanley_control(self, path_x, path_y):
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
            A ** 2 + B ** 2)  # Distance from a point to a line in the plane (meters)
        theta_d = np.arctan2(Ke * error_front_axle, Kv + self.v)

        # Para saber el error maximo que se da en el recorrido
        if self.error_maximo < error_front_axle:
            self.error_maximo = error_front_axle

        # Steering control
        delta = theta_e + theta_d
        self.steering_angle = np.clip(delta, -max_steer, max_steer)

        self.error.append(error_front_axle)

        # Calculate RMS error (RMS = raiz((1/n)*sum(error^2)))
        self.rmse_acumulado = np.sqrt(np.mean(np.square(np.array(self.error))))

        # print("error_cuadrado: {}".format(self.error_cuadrado))
        # print("suma error cuadrado: {}".format(sum(self.error_cuadrado)))
        # print("RMS: {}".format(self.rmse_acumulado))

    def calc_target_spline(self, path_x, path_y):
        """
        Calculate the index of the closest point of the path
        :param path_x: [float] x coordinates list of the path planning
        :param path_y: [float] y coordinates list of the path planning
        :return: (int, float)  Index of the way point at the shortest distance
        """
        min_idx = 0
        min_dist = float("inf")
        global indice_prev
        global target_speed

        # Search nearest waypoint
        for i in range(len(path_x)):
            dist = np.linalg.norm(np.array([path_x[i] - self.x, path_y[i] - self.y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        
        
        if (min_idx - indice_prev)<0 and vuelta_reconomiento==1 :
            self.flag_entrada_circulo=1
            
        if (self.flag_entrada_circulo==1 and np.linalg.norm(np.array([path_x[0] - self.x, path_y[0] - self.y]))>6 ):
            self.contador_de_vuelta=self.contador_de_vuelta+1
            print("se ha dado una vuelta contador = %d " % self.contador_de_vuelta )
            self.flag_entrada_circulo=0

        if self.contador_de_vuelta==10:
            target_speed=0
            self.finish_flag=1
        indice_prev=min_idx

        return min_idx

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

