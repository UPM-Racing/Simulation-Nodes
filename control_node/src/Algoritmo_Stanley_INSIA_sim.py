#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import rospy
import math
import pandas
import argparse
import time
import scipy.interpolate as scipy_interpolate
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64



# Variables to calculate the error made along the path
error_u = 0
error_list = []
error_maximo = 0.0
contador = 0
mean_deviation = 0

Ke = 0.3  # control gain
Kv = 10 # Ganancia que evita que el error se vaya a infinito cuando la velocidad es cero
Kp = 1.0 # speed propotional gain
ki = 1.0 # speed integral gain
kd = 0.1 # speed derivational gain
dist_yaw = 0.3 # distancia cada cuanto se actualiza yaw
dt = 0.1  # [s] time difference 1/frecuencia GPS
L = 2.55  # [m] Wheel base of vehicle
max_steer = 1.22  # [rad] max steering angle
vel_obj_kmh = 10.0 # velocidad a la que llegaria el control PID del vehiculo
k_proxima = 0.001 # Distancia para la que se considera haber llegado al ultimo target
freq = 10 # 10hz
pub = 0	#publisher node variable as global
show_animation = True
gps = [0, 0]
target_speed = 10.0 / 3.6     
max_accel = 1.0 

class State(object):
	"""
	Class representing the state of a vehicle.
	:param x: (float) x-coordinate
	:param y: (float) y-coordinate
	:param yaw: (float) yaw angle
	:param v: (float) speed
	"""

	def __init__(self):
		#super(State, self).__init__()
		self.v = 0.0
		self.x = 0.0
		self.y = 0.0
		self.yaw = 0.0
		self.n_course_point = 200
		self.degree = 3
		
		
		
		#self.steering_real
		self.steering_angle=0.0
		self.acceleration = 0.0
		self.last_idx = 0

		#Bandera para comprobar si ya se ha definido el primer target
		self.flag_target_init = 0



		#Condicion para la aproximacion del yaw en caso de utilizar el metodo con atan2
		#self.prev_p = [x,y]
		#self.delta_vec = [0.0]
		#self.prev_angle = 0



	
	def update_velocity(self, v):
		self.v = v






	def update_car_position(self, pose):		
		self.x = pose.position.x
		self.y = pose.position.y
		self.yaw = self.normalize_angle(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))
		print (self.x ,self.y)
		

	def pid_control(self, target):
		accel = Kp * (target - self.v) + ki * (target - self.v) * dt + kd * (
				target - self.v) / dt
		self.acceleration = np.clip(accel, 0, max_accel)        # Freno por inercia, maxima aceleracion 1 m/s^2

		# print("Aceleracion: {}".format(accel))
		#return accel
		
	def stanley_control(self, cx_short, cy_short):
		"""
		Stanley steering control.
		:param state: (State object)
		:param cx: ([float])
		:param cy: ([float])
		:return: (float, int)
		"""
		global error_u, contador
		contador = contador + 1
		current_target_idx = self.calc_target_spline( cx_short, cy_short)
		if current_target_idx < (len(cx_short) - 1):
			diff_x = cx_short[current_target_idx + 1] - cx_short[current_target_idx]
			diff_y = cy_short[current_target_idx + 1] - cy_short[current_target_idx]
		else:
			diff_x = cx_short[current_target_idx] - cx_short[current_target_idx - 1]
			diff_y = cy_short[current_target_idx] - cy_short[current_target_idx - 1]
		yaw_path = np.arctan2(diff_y, diff_x)

		# theta_e corrects the heading error

		theta_e = self.normalize_angle(yaw_path - self.yaw)

		# theta_d corrects the cross track error
		if current_target_idx < (len(cx_short) - 1):
			A = cy_short[current_target_idx + 1] - cy_short[current_target_idx]
			B = cx_short[current_target_idx] - cx_short[current_target_idx + 1]
			C = cy_short[current_target_idx] * cx_short[current_target_idx + 1] - cx_short[current_target_idx] * cy_short[
				current_target_idx + 1]
		else:
			A = cy_short[current_target_idx] - cy_short[current_target_idx - 1]
			B = cx_short[current_target_idx - 1] - cx_short[current_target_idx]
			C = cy_short[current_target_idx - 1] * cx_short[current_target_idx] - cx_short[current_target_idx - 1] * \
				cy_short[current_target_idx]

		error_front_axle = (A * self.x + B * self.y + C) / np.sqrt(A ** 2 + B ** 2)
		theta_d = np.arctan2(Ke * error_front_axle, Kv + self.v)

		# Steering control
		delta = theta_e + theta_d
		self.steering_angle  = np.clip(delta, -max_steer, max_steer)
		"""print(ecm_acumulado)
		print("Ecm_acumulado {}".format(ecm_acumulado))
		print(delta)
		print("Delta(degrees): {}".format(np.degrees(delta)))"""

		return delta, current_target_idx

	def normalize_angle(self, angle):
		"""
		Normalize an angle to [-pi, pi].
		:param angle: (float)
		:return: (float) Angle in radian in [-pi, pi]
		"""
		return (angle + math.pi) % (2 * math.pi) - math.pi


	def calc_target_cono(self, ax, ay, current_idx):
		"""
		Compute index in the trajectory list of the target.
		:param state: (State object)
		:param cx: [float]
		:param cy: [float]
		:return: (int, float)
		"""

		# En el coche del insia el gps esta en el eje delantero

		if self.flag_target_init == 1:
			target_idx = current_idx + 20

		else:
			target_idx = 0
			min_dist = float("inf")

			# Search nearest waypoint
			for i in range(len(ax)):
				dist = np.linalg.norm(np.array([ax[i] - self.x, ay[i] - self.y]))
				if dist < min_dist:
					min_dist = dist
					target_idx = i
			

		return target_idx


	def calc_target_spline(self, path_x, path_y):
		"""
		Calculate the index of the closest point of the path
		:param path_x: [float] x coordinates list of the path planning
		:param path_y: [float] y coordinates list of the path planning
		:return: (int, float)  Index of the way point at the shortest distance
		"""
		min_idx = 0
		min_dist = float("inf")

		# Search nearest waypoint
		for i in range(len(path_x)):
			dist = np.linalg.norm(np.array([path_x[i] - self.x, path_y[i] - self.y]))
			if dist < min_dist:
				min_dist = dist
				min_idx = i
		return min_idx


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

	def calc_vector_spline(self, ax, ay, cono_target_idx):
		"""
			Calculate vector spline knowing 10 meters of the path. (each point is separated 0.5 meters aprox)
			from the next.
			:param state: (State object)
			:param ax: [float]
			:param ay: [float]
			:param cono_target_idx: [int]
			:return: (float, float)
			"""

		if (len(ax)-cono_target_idx) < 20 :
			aux=len(ax)-cono_target_idx
		else :
			aux=20
		aux_x = np.zeros(aux)
		aux_y = np.zeros(aux)



		i = 0
		t = cono_target_idx

		while i < aux:
			aux_x[i] = ax[t]
			aux_y[i] = ay[t]
			i += 1
			t += 1

		cx_short, cy_short = 	self.interpolate_b_spline_path(aux_x , aux_y, 200, 3)
		return cx_short, cy_short





	def principal_loop(self,path_planning):
		poses = path_planning.poses

		
		ax = []
		ay = []
		cx_short=[]
		cy_short=[]
		spline_target_idx = 0
		#cyaw_short=[]

		for i, pose in enumerate(poses):
			ax.append(pose.pose.position.x)
			ay.append(pose.pose.position.y)
		#path_x, path_y = self.interpolate_b_spline_path(ax, ay, self.n_course_point, self.degree)
		#trabajo con ax y ay 


		

		
		cono_target_idx = 0
		cono_target_idx = self.calc_target_cono( ax, ay, cono_target_idx)
		cx_short, cy_short = self.calc_vector_spline (ax, ay, cono_target_idx)
		self.flag_target_init = 1#pruebar inicar la flag cuando se reciva el gps
		


		while (not rospy.is_shutdown()) :
			print ('dentro del while ')
			self.pid_control(target_speed)

			if (self.v > 0):
				if (spline_target_idx == (len(cx_short)-1)):
					if abs(len(ax)-cono_target_idx) < 20:
						print('frenas')#state.flag_frenado = 1
					else:
						cono_target_idx = self.calc_target_cono( ax, ay, cono_target_idx)
						cx_short, cy_short = self.calc_vector_spline(ax, ay, cono_target_idx)
						di, spline_target_idx = self.stanley_control(cx_short, cy_short)

				else:
					di, spline_target_idx = self.stanley_control( cx_short, cy_short)

				#print("Cono target idx {}/{}".format(cono_target_idx, self.last_idx))
				#print("Spline target idx {}/{}".format(spline_target_idx, len(cx_short)-1))

			else:
				di=0
				#print('LA VELOCIDAD ES NULA')









