#! /usr/bin/env python
import rospy
import math
import numpy as np
from scipy.spatial import distance
from eufs_msgs.msg import ConeArrayWithCovariance, ConeWithCovariance, CarState
from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion, Vector3Stamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from eufs_msgs.msg import WheelSpeedsStamped
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped

Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2

N_PARTICLE = 50  # number of particle
NTH = N_PARTICLE / 1.3  # Number of particle for re-sampling

class Particle:
    def __init__(self, n_landmark):
        self.w = 1.0 / N_PARTICLE
        # state variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.P = np.eye(3)
        # landmark x-y positions with shape (n_landmark, LM_SIZE)
        self.lm = np.zeros((n_landmark, 2))
        # landmark position covariance
        self.lmP = np.zeros((n_landmark * 2, 2))

    def expand_particle(self, number_particles):
        v_add = np.zeros((number_particles, 2))
        v_add_cov = np.zeros((number_particles*2, 2))

        self.lm = np.vstack((self.lm, v_add))
        self.lmP = np.vstack((self.lmP, v_add_cov))


class FastSLAM:
    def __init__(self):
        self.past_time = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.DT = 0.0
        self.THRESHOLD = 1.5
        self.marker_ests = MarkerArray()
        self.slam_marker_ests = MarkerArray()
        self.total_landmarks = 0
        self.u = np.zeros((2, 1))
        self.path = Path()
        self.pose = PoseStamped()
        self.wheelbase = 1.58
        self.state_x = 0.0
        self.state_y = 0.0
        self.state_yaw = 0.0

        # se crean las particulas vacias para que se expandan con la llegada de landmarks
        self.particles = [Particle(0) for _ in range(N_PARTICLE)]

    # por cada landmark de entrada:
    def bucle_principal(self, landmarks, time):
        # los landmarks es una lista de objetos de landmark que poseen posicion
        self.DT = time - self.past_time
        self.past_time = time

        #calcular las posiciones medias de los conos en las particulas
        landmarks_x, landmarks_y = self.calc_final_lm_position(self.particles)

        z = np.zeros((3, 0))

        cones_x = []
        cones_y = []

        for i, cone in enumerate(landmarks):
            if len(landmarks_x) == 0:
                index = self.total_landmarks
                self.expandir_particulas(1)
                self.total_landmarks += 1
            else:
                ''' Se calcula a partir de la posicion del coche anterior, igual habria que hacer el motion model antes'''
                x, y = self.local_to_global(cone.point.x, cone.point.y)
                cones_x.append(x)
                cones_y.append(y)
                new_landmark_position = np.array([(x, y)])
                landmarks_x = landmarks_x.reshape(-1, 1)
                landmarks_y = landmarks_y.reshape(-1, 1)
                landmark_positions = np.hstack((landmarks_x, landmarks_y))
                diferencias = distance.cdist(new_landmark_position, landmark_positions, 'euclidean')
                diferencias = diferencias.reshape(-1,)
                index_min = np.argmin(diferencias)

                #print(diferencias[index_min])
                #print('------')

                if diferencias[index_min] <= self.THRESHOLD:
                    index = index_min
                else:
                    index = self.total_landmarks
                    self.expandir_particulas(1)
                    self.total_landmarks += 1

            d, angle = self.definir_posicion(cone.point.x, cone.point.y)
            zi = np.array([d, self.pi_2_pi(angle), index]).reshape(3, 1)
            z = np.hstack((z, zi))

        #print(self.total_landmarks)
        #print('-----------------------------------')

        self.particles = self.fast_slam2(self.particles, z)
        xEst = self.calc_final_state(self.particles)

        self.x = xEst[0, 0]
        self.y = xEst[1, 0]
        self.yaw = xEst[2, 0]

        marker_x_high = []
        marker_y_high = []
        marker_x_low = []
        marker_y_low = []
        for particle in self.particles:
            if particle.w > (0.9 / N_PARTICLE):
                marker_x_high.append(particle.x)
                marker_y_high.append(particle.y)
            else:
                marker_x_low.append(particle.x)
                marker_y_low.append(particle.y)

        self.marker_array(marker_x_high, marker_y_high, marker_x_low, marker_y_low)
        self.marker_array_slam()
        self.publish_path()

    def expandir_particulas(self, number_new_particles):
        for i in range(N_PARTICLE):
            self.particles[i].expand_particle(number_new_particles)

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

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def fast_slam2(self, particles, z):
        """
        :param particles: list of Particle objects
        :param z: the observations detected
        :return:
        """
        particles = self.predict_particles(particles)
        particles = self.update_with_observation(particles, z)
        particles = self.resampling(particles)

        return particles

    def predict_particles(self, particles):
        """
        :param particles: list of Particle objects
        :return: list of Particle objects with updated state vectors
        """
        # para cada particula
        for i in range(N_PARTICLE):
            # se extrae el estado de la particula y se almacena en px
            px = np.zeros((3, 1))
            px[0, 0] = particles[i].x
            px[1, 0] = particles[i].y
            px[2, 0] = particles[i].yaw
            # se vuelve a anadir ruido a el control input vector
            # (2,1) u = [vel, yaw_rate]
            ud = self.u + (np.matmul(np.random.randn(1, 2) / 4.0, R ** 0.5)).T  # add noise
            # se actualiza el estado de las particulas en funcion del motion model
            px = self.motion_model(px, ud)
            particles[i].x = px[0, 0]
            particles[i].y = px[1, 0]
            particles[i].yaw = px[2, 0]

        return particles

    def update_with_observation(self, particles, z):
        """
        :param particles: list of Particle objects with updated state vectors
        :param z: the observations detected
        :return: list of Particle objects with updated state vectors and landmarks
        """
        # Por cada landmark en z
        for iz in range(len(z[0, :])):
            landmark_id = int(z[2, iz])

            # por cada particula
            for ip in range(N_PARTICLE):
                # new landmark
                # si la x es 0 se anade un landmark
                if abs(particles[ip].lm[landmark_id, 0]) == 0:
                    particles[ip] = self.add_new_landmark(particles[ip], z[:, iz], Q)
                # known landmark
                else:
                    w = self.compute_weight(particles[ip], z[:, iz], Q)
                    # Se multiplica el peso por el guardado
                    particles[ip].w *= w
                    particles[ip] = self.update_landmark(particles[ip], z[:, iz], Q)
                    particles[ip] = self.proposal_sampling(particles[ip], z[:, iz], Q)

        return particles

    def proposal_sampling(self, particle, z, Q_cov):
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
        # State
        x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
        P = particle.P
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        Sfi = np.linalg.inv(Sf)
        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        Pi = np.linalg.inv(P)

        particle.P = np.linalg.inv(np.matmul(np.matmul(Hv.T, Sfi), Hv) + Pi)  # proposal covariance
        x += np.matmul(np.matmul(np.matmul(particle.P, Hv.T), Sfi), dz)  # proposal mean

        particle.x = x[0, 0]
        particle.y = x[1, 0]
        particle.yaw = x[2, 0]

        return particle

    def add_new_landmark(self, particle, z, Q_cov):
        """
        :param particle: Particle object with updated state vectors
        :param z: one observation [distance, angle, landmark_id]
        :param Q_cov: Q
        :return: Particle object with updated state vectors and covariances
        """
        # extraemos la distancia, el angulo y el landmark id
        r = z[0]
        b = z[1]
        lm_id = int(z[2])

        # calculamos el seno y el coseno del angulo del coche mas el de la observacion
        s = math.sin(self.pi_2_pi(particle.yaw + b))
        c = math.cos(self.pi_2_pi(particle.yaw + b))

        # se anaden la x y la y correspondientes a la lista de landmarks
        particle.lm[lm_id, 0] = particle.x + r * c
        particle.lm[lm_id, 1] = particle.y + r * s

        # covariance
        # calculamos las distancias al coche de la landmark
        dx = r * c
        dy = r * s
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        # es el jacobiano de la posicion de los landmarks
        Gz = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])
        particle.lmP[2 * lm_id:2 * lm_id + 2] = np.matmul(np.matmul(np.linalg.inv(Gz), Q_cov), np.linalg.inv(Gz.T))

        return particle

    def compute_weight(self, particle, z, Q_cov):
        """
        :param particle: Particle object with updated state vectors
        :param z: one observation [distance, angle, landmark_id]
        :param Q_cov: Q
        :return: the w value for the particle
        """
        # obtiene el landmark id y la x y la y guardadas en lm
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        # saca la covarianza
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        # La diferencia entre z y zp
        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        # Calculamos la inversa de Sf
        try:
            invS = np.linalg.inv(Sf)
        except np.linalg.linalg.LinAlgError:
            print("singular")
            return 1.0

        # Se calcula el peso de la particula con la formula
        num = math.exp(-0.5 * np.matmul(np.matmul(dz.T, invS), dz))
        den = math.sqrt(np.linalg.det(2.0 * math.pi * Sf))

        w = num / den

        return w

    def compute_jacobians(self, particle, xf, Pf, Q_cov):
        """
        :param particle: Particle object with updated state vectors
        :param xf: stored x and y values
        :param Pf: stored covariance matrix
        :param Q_cov: Q
        :return:
            zp: [distance, angle in global axis]
            Hv: jacobian of the car state vector
            Hf: jacobian of the landmarks
            Sf: measurement covariance
        """
        # calculamos las distancias al coche de la landmark
        dx = xf[0, 0] - particle.x
        dy = xf[1, 0] - particle.y
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        # calculamos el angulo del coche a la landmark en ejes globales
        zp = np.array([d, self.pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

        # es el jacobiano de el vector del coche
        Hv = np.array([[-dx / d, -dy / d, 0.0],
                       [dy / d2, -dx / d2, -1.0]])

        # es el jacobiano de la posicion de los landmarks
        Hf = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])

        Sf = np.matmul(np.matmul(Hf, Pf), Hf.T) + Q_cov

        return zp, Hv, Hf, Sf

    def update_landmark(self, particle, z, Q_cov):
        """
        :param particle: Particle object with updated state vectors
        :param z: one observation [distance, angle, landmark_id]
        :param Q_cov: Q
        :return: Particle object with updated state vectors and landmark
        """
        # obtiene el landmark id y la x y la y guardadas en lm
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        # Saca la covarianza
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])

        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        # La diferencia entre z y zp
        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        xf, Pf = self.update_kf_with_cholesky(xf, Pf, dz, Q, Hf)

        particle.lm[lm_id, :] = xf.T
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

        return particle

    def update_kf_with_cholesky(self, xf, Pf, v, Q_cov, Hf):
        PHt = np.matmul(Pf, Hf.T)
        S = np.matmul(Hf, PHt) + Q_cov

        S = (S + S.T) * 0.5
        SChol = np.linalg.cholesky(S).T
        SCholInv = np.linalg.inv(SChol)
        W1 = np.matmul(PHt, SCholInv)
        W = np.matmul(W1, SCholInv.T)

        x = xf + np.matmul(W, v)
        P = Pf - np.matmul(W1, W1.T)

        return x, P

    def resampling(self, particles):
        """
        low variance re-sampling
        """

        particles = self.normalize_weight(particles)

        pw = []
        for i in range(N_PARTICLE):
            pw.append(particles[i].w)

        # array con todos los pesos de las particulas
        pw = np.array(pw)

        n_eff = 1.0 / (np.matmul(pw, pw.T))  # Effective particle number
        # print(n_eff)

        # resampling when n_eff drops below threshold
        if n_eff < NTH:  # resampling
            # Return the cumulative sum of the elements along a given axis
            w_cum = np.cumsum(pw)
            # vector tipo [0, 0.01, 0.02, ...., 0.99]
            base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
            # A la base se le suma un numero random dividido por 100
            resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

            inds = []
            ind = 0
            # se recorren todos los pesos y mientras que el resample sea mayor que el w_cum se introduce ese indice en los inds
            for ip in range(N_PARTICLE):
                while (ind < w_cum.shape[0] - 1) and (resample_id[ip] > w_cum[ind]):
                    ind += 1
                inds.append(ind)

            # Recrea todas las particulas en funcion de los indices almacenados en inds
            tmp_particles = particles[:]
            for i in range(len(inds)):
                particles[i].x = tmp_particles[inds[i]].x
                particles[i].y = tmp_particles[inds[i]].y
                particles[i].yaw = tmp_particles[inds[i]].yaw
                particles[i].lm = tmp_particles[inds[i]].lm[:, :]
                particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
                particles[i].w = 1.0 / N_PARTICLE

        return particles

    def normalize_weight(self, particles):
        """
        :param particles: list of Particle objects with updated state vectors
        :return:
        """
        # Suma todos los pesos de las particulas
        sum_w = sum([p.w for p in particles])

        # Divide el peso de cada particula por la suma
        try:
            for i in range(N_PARTICLE):
                particles[i].w /= sum_w
        except ZeroDivisionError:
            for i in range(N_PARTICLE):
                particles[i].w = 1.0 / N_PARTICLE

            return particles

        return particles

    def calc_final_state(self, particles):
        xEst = np.zeros((3, 1))

        particles = self.normalize_weight(particles)

        # media de todas las posiciones con sus pesos
        for i in range(N_PARTICLE):
            xEst[0, 0] += particles[i].w * particles[i].x
            xEst[1, 0] += particles[i].w * particles[i].y
            xEst[2, 0] += particles[i].w * particles[i].yaw

        xEst[2, 0] = self.pi_2_pi(xEst[2, 0])
        #  print(xEst)

        return xEst

    def calc_final_lm_position(self, particles):
        landmarks_x = []
        landmarks_y = []

        particles = self.normalize_weight(particles)

        # media de todas las posiciones con sus pesos
        for j in range(self.total_landmarks):
            xEst = 0.0
            yEst = 0.0
            for i in range(N_PARTICLE):
                xEst += particles[i].w * particles[i].lm[j, 0]
                yEst += particles[i].w * particles[i].lm[j, 1]
            landmarks_x.append(xEst)
            landmarks_y.append(yEst)

        return np.array(landmarks_x), np.array(landmarks_y)

    def update_car_position(self, pose):
        self.state_x = pose.position.x
        self.state_y = pose.position.y
        self.state_yaw = self.pi_2_pi(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))

    def update_car_position_odom(self, time, velocity, yaw_rate):
        self.u[0, 0] = velocity
        self.u[1, 0] = yaw_rate

    def update_car_yaw(self, yaw):
        self.u[1, 0] = yaw

    def update_car_gps_vel(self, velocity):
        self.u[0, 0] = velocity

    def local_to_global(self, x, y):
        position = np.array([x, y])
        C_ns = np.array([[np.cos(self.state_yaw), -np.sin(self.state_yaw)], [np.sin(self.state_yaw), np.cos(self.state_yaw)]])
        rotated_position = C_ns.dot(position) + np.array([self.state_x, self.state_y])

        '''d, angle = self.definir_posicion(x, y)

        # calculamos el seno y el coseno del angulo del coche mas el de la observacion
        s = math.sin(self.pi_2_pi(self.yaw + self.pi_2_pi(angle)))
        c = math.cos(self.pi_2_pi(self.yaw + self.pi_2_pi(angle)))

        # se anaden la x y la y correspondientes a la lista de landmarks
        x = self.x + d * c
        y = self.y + d * s'''

        return rotated_position[0], rotated_position[1]

    def marker_array(self, cones_x, cones_y, cones_x_low, cones_y_low):
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
            point.z = 0.2
            pose.position = point
            orientation = Quaternion()
            # Suponiendo roll y pitch = 0
            orientation.x = 0.0
            orientation.y = 0.0
            orientation.z = 0.0
            orientation.w = 1.0
            pose.orientation = orientation
            marker_est.pose = pose
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (0, 0, 255)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.1, 0.1, 0.4)
            self.marker_ests.markers.append(marker_est)

        for i in range(len(cones_x_low)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = cones_x_low[i]
            point.y = cones_y_low[i]
            point.z = 0.2
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
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.1, 0.1, 0.4)
            self.marker_ests.markers.append(marker_est)

    def marker_array_slam(self):
        # Publish it as a marker in rviz
        self.slam_marker_ests.markers = []

        # calcular las posiciones medias de los conos en las particulas
        distances_x, distances_y = self.calc_final_lm_position(self.particles)

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
            self.slam_marker_ests.markers.append(marker_est)

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
        pose.pose.orientation = orientation
        self.path.poses.append(pose)
        self.pose = pose


class Slam_Class(object):
    def __init__(self):
        self.fast_slam = FastSLAM()
        self.cones_sub = rospy.Subscriber('/ground_truth/cones', ConeArrayWithCovariance, self.cones_callback)

        self.slam_marker_array_pub = rospy.Publisher('/slam_marker_array_pub', MarkerArray, queue_size=1)
        self.marker_array_pub = rospy.Publisher('/particles_marker_array_pub', MarkerArray, queue_size=1)

        # Solo pueden estar conectado 1 de estos dos siguientes
        self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)
        # self.state_estimation_sub = rospy.Subscriber('/pose_pub', PoseStamped, self.state_estimation_callback)

        # Solo pueden estar conectados odom, o imu y gps
        # self.odom_sub = rospy.Subscriber('/odometry_pub', WheelSpeedsStamped, self.odom_callback)
        # self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)

        self.path_pub = rospy.Publisher('/slam_path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/slam_pose_pub', PoseStamped, queue_size=1)
        self.first = 2

    def sub_callback(self, msg):
        self.fast_slam.update_car_position(msg.pose.pose)

    def state_estimation_callback(self, msg):
        self.fast_slam.update_car_position(msg.pose)

    def cones_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        cones = msg.yellow_cones + msg.blue_cones + msg.big_orange_cones
        self.fast_slam.bucle_principal(cones, timestamp)

    def odom_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity_rpm = msg.lb_speed
        velocity_mean = (velocity_rpm * 2.0 * math.pi * 0.25) / 60.0

        # Max steering = 0.52 rad/s
        steering = msg.steering

        self.fast_slam.update_car_position_odom(timestamp, velocity_mean, steering)

    def gps_vel_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        velocity = [-msg.vector.y + 1*10**-12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        self.fast_slam.update_car_gps_vel(velocity_mean)

    def imu_callback(self, msg):
        yaw_rate = msg.angular_velocity.z
        self.fast_slam.update_car_yaw(yaw_rate)

    def control_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        time = time_sec + time_nsec
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration

        self.fast_slam.update_car_yaw(angle)


if __name__ == '__main__':
    rospy.init_node('slam_node', anonymous=True)
    slam_class = Slam_Class()
    #rospy.spin()
    rate = rospy.Rate(10)  # Hz

    while not rospy.is_shutdown():
        slam_class.marker_array_pub.publish(slam_class.fast_slam.marker_ests)
        slam_class.slam_marker_array_pub.publish(slam_class.fast_slam.slam_marker_ests)
        slam_class.path_pub.publish(slam_class.fast_slam.path)
        slam_class.pose_pub.publish(slam_class.fast_slam.pose)
        rate.sleep()