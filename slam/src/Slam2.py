#! /usr/bin/env python
import rospy
import math
import numpy as np
from scipy.spatial import distance
import time
import csv

from geometry_msgs.msg import Pose, Point, PoseStamped, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker


WITH_EKF = True

Q = np.diag([0.3, np.deg2rad(5.0)]) ** 2
R = np.diag([0.5, np.deg2rad(30.0)]) ** 2

N_PARTICLE = 50  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

class Gps():
    def __init__(self):
        # Constantes de transformacion del gps
        self.a = 6378137
        self.b = 6356752.3142
        self.f = (self.a - self.b) / self.a
        self.e_sq = self.f * (2 - self.f)

        # Inicializacion de variables del gps
        self.LONGITUD_0 = 0
        self.LATITUD_0 = 0
        self.ALTURA_0 = 0

    def geodetic_to_ecef(self, lat, lon, h):
        '''
        Pasar de coordenadar geodetic a ecef.

        :param lat: latitud
        :param lon: Longitud
        :param h: Altura
        :return: x, y, z locales
        '''
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
        '''
        Pasar de coordenadas ecef a enu.

        :param x: x local
        :param y: y local
        :param z: z local
        :param lat0: Latitud inicial
        :param lon0: Longitud inicial
        :param h0: Altura inicial
        :return: x, y, z (en este caso solo x, y) locales
        '''
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
        '''
        Pasar de coordenadas geodetic a enu.

        :param lat: Latitud
        :param lon: Longitud
        :param h: Altura
        :param lat_ref: Latitud inicial
        :param lon_ref: Longitud inicial
        :param h_ref: Altura inicial
        :return: x, y, z (en este caso solo x, y) locales
        '''
        x, y, z = self.geodetic_to_ecef(lat, lon, h)

        return self.ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

    def gps_to_local(self, latitude, longitude, altitude):
        '''
        Funcion que devuelve en ejes x, y, z locales (en este caso solo x e y) a partir de la latitud, longitud y
        altura.

        :param latitude: Valor de latitud del gps
        :param longitude: Valor de longitud del gps
        :param altitude: Valor de altitud del gps
        :return: Coordenadas del coche en un plano local centrado en el primer dato de latitud, longitud y altura
        '''
        if self.LONGITUD_0 == 0:
            self.LONGITUD_0 = longitude
            self.LATITUD_0 = latitude
            self.ALTURA_0 = altitude

        x, y = self.geodetic_to_enu(latitude, longitude, altitude, self.LATITUD_0, self.LONGITUD_0, self.ALTURA_0)

        return x, y


class Particle():
    def __init__(self, n_landmark):
        # Estado del coche y landmarks en la particula
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lm = np.zeros((n_landmark, 2)) # Posicion x-y de cada landmark

        # Variables para los calculos
        self.w = 1.0 / N_PARTICLE # Peso de la particula
        self.P = np.eye(3)
        self.lmP = np.zeros((n_landmark * 2, 2)) # Covarianza de la posicion de cada landmark
        if WITH_EKF:
            self.p_cov = np.eye(3) # Covarianza de la posicion del coche

    def expand_particle(self, number_particles):
        '''
        Anade a la particula un nuevo landmark.

        :param number_particles: Numero de landmarks a anadir
        '''
        v_add = np.zeros((number_particles, 2))
        v_add_cov = np.zeros((number_particles*2, 2))

        self.lm = np.vstack((self.lm, v_add))
        self.lmP = np.vstack((self.lmP, v_add_cov))


class SLAM():
    def __init__(self):
        ''' Variables no configurables '''
        # Estado del coche
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Inicializacion de variables
        self.past_time = 0.0
        self.DT = 0.0
        self.u = np.zeros((2, 1))
        self.total_landmarks = 0
        self.path = Path()
        self.pose = PoseStamped()
        self.marker_ests = MarkerArray()
        self.slam_marker_ests = MarkerArray()
        self.h_jac = np.zeros([2, 3])
        self.h_jac[:, :2] = np.eye(2)  # Jacobiano del modelo cinematico
        self.cones_to_csv = False

        # Variables no usadas en principio
        self.state_x = 0.0
        self.state_y = 0.0
        self.state_yaw = 0.0

        # Se crean las particulas vacias para que se expandan con la llegada de landmarks
        self.particles = [Particle(0) for _ in range(N_PARTICLE)]

        ''' Variables configurables '''
        # Datos del coche
        self.radio = 0.2525  # Radio de la rueda del coche
        self.wheelbase = 1.58  # Distancia entre ejes del coche
        self.lr = 0.711  # Distancia del centro de gravedad al eje trasero del coche

        # Constantes para Fast SLAM 2.0
        self.THRESHOLD = 1.5
        self.var_gnss = 0.25
        self.var_model = 0.1

    def bucle_principal(self, landmarks, timestamp):
        '''
        Bucle principal de actualizacion de las particulas. Se llama cada vez que llega un mensaje de conos. Comprueba
        si cada observacion corresponde a un cono ya guardado o es uno nuevo antes de llamar a la funcion de slam.

        :param landmarks: Lista de la posicion de todos los conos observables
        :param time: Instante de tiempo actual
        '''
        '''period = time.time()'''
        self.DT = timestamp - self.past_time
        self.past_time = timestamp

        # Calcular las posiciones medias de los conos en las particulas
        landmarks_x, landmarks_y = self.calc_final_lm_position(self.particles)

        z = np.zeros((3, 0))
        # cones_x = []
        # cones_y = []

        for i, cone in enumerate(landmarks):
            # Cuando no hay landmarks todavia (al inicio)
            if len(landmarks_x) == 0:
                index = self.total_landmarks
                self.expandir_particulas(1)
                self.total_landmarks += 1
            else:
                x, y = self.local_to_global(cone.point.x, cone.point.y)
                # cones_x.append(x)
                # cones_y.append(y)
                new_landmark_position = np.array([(x, y)])
                landmarks_x = landmarks_x.reshape(-1, 1)
                landmarks_y = landmarks_y.reshape(-1, 1)
                landmark_positions = np.hstack((landmarks_x, landmarks_y))
                diferencias = distance.cdist(new_landmark_position, landmark_positions, 'euclidean')
                diferencias = diferencias.reshape(-1,)
                index_min = np.argmin(diferencias)

                # Si dos conos estan mas cerca que el threshold, son el mismo cono
                if diferencias[index_min] <= self.THRESHOLD:
                    index = index_min
                else:

                    index = self.total_landmarks
                    self.expandir_particulas(1)
                    self.total_landmarks += 1

            # Obtenemos la distancia y angulo en coordenadas locales a partir de la observacion
            d, angle = self.definir_posicion(cone.point.x, cone.point.y)
            zi = np.array([d, self.pi_2_pi(angle), index]).reshape(3, 1)
            z = np.hstack((z, zi))

        self.particles = self.fast_slam2(self.particles, z)
        xEst = self.get_best_position(self.particles)

        self.x = xEst[0, 0]
        self.y = xEst[1, 0]
        self.yaw = xEst[2, 0]

        '''period = time.time() - period
        with open('../catkin_ws/results/Slam2.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
            writer.writerow([period])'''

        # Actualiza las variables de Rviz
        self.rviz_update()

        '''if self.total_landmarks == 71 and not self.cones_to_csv:
            with open('../catkin_ws/results/cones_position_slam2.csv', 'ab') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
                ind = self.get_best_particle(self.particles)
                for i, landmark in enumerate(self.particles[ind].lm):
                    writer.writerow([i, landmark[0], landmark[1]])
                print('done')
                self.cones_to_csv = True'''

    def expandir_particulas(self, number_new_particles):
        '''
        Extiende el vector que contiene la posicion de landmarks y sus covarianzas de cada particula.

        :param number_new_particles: Numero de landmarks nuevos a anadir
        '''
        for i in range(N_PARTICLE):
            self.particles[i].expand_particle(number_new_particles)

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

    def motion_model(self, x, u, p_cov=0):
        '''
        Modelo cinematico basado en el kinematic bicycle model del coche. Se pasa para cada particula.

        :param x: Estado anterior de la particula
        :param u: Entrada al modelo cinematico (velocidad y giro de volante)
        :param p_cov: Matriz de covarianzas de la particula
        :return: Estado nuevo y nueva matriz de covarianzas de la particula
        '''
        beta = math.atan2(self.lr * math.tan(u[1, 0]), self.wheelbase)

        x_dot = u[0, 0] * math.cos(x[2, 0] + beta)
        y_dot = u[0, 0] * math.sin(x[2, 0] + beta)
        yaw_dot = u[0, 0] * math.sin(beta) / self.lr

        x[0, 0] = x[0, 0] + x_dot * self.DT
        x[1, 0] = x[1, 0] + y_dot * self.DT
        x[2, 0] = x[2, 0] + yaw_dot * self.DT

        x[2, 0] = self.pi_2_pi(x[2, 0])

        if WITH_EKF:
            F = np.array([[1, 0, -u[0, 0] * math.sin(x[2, 0]) * self.DT],
                          [0, 1, u[0, 0] * math.cos(x[2, 0]) * self.DT],
                          [0, 0, 1]])

            Qi = np.eye(3)
            Qi[0:2, 0:2] = self.var_model * Qi[0:2, 0:2]
            Qi[4:5, 4:5] = self.var_model * Qi[2:3, 2:3]
            Qi = (self.DT ** 2) * Qi

            # Propagar incertidumbre
            p_cov = F.dot(p_cov).dot(F.T) + Qi

            return x, p_cov
        else:
            return x

    def pi_2_pi(self, angle):
        '''
        Devuelve el angulo en un rango entre -pi y pi.

        :param angle: Angulo de entrada
        :return: Angulo pasado al rango [-pi, pi]
        '''
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def fast_slam2(self, particles, z):
        """
        Funcion global de Fast SLAM 2.0.

        :param particles: Lista de todas las particulas
        :param z: Observaciones detectadas
        :return: Particulas actualizadas
        """
        particles = self.predict_particles(particles)
        particles = self.update_with_observation(particles, z)
        particles = self.resampling(particles)

        return particles

    def predict_particles(self, particles):
        """
        Actualiza el estado de cada particula pasandola por el motion model.

        :param particles: Lista de todas las particulas
        :return: Lista de las particulas con los vectores de estado actualizados
        """
        # Para cada particula
        for i in range(N_PARTICLE):
            # Se extrae el estado de la particula y se almacena en px
            px = np.zeros((3, 1))
            px[0, 0] = particles[i].x
            px[1, 0] = particles[i].y
            px[2, 0] = particles[i].yaw
            # Se vuelve a anadir ruido al control input vector para generar particulas diferentes entre si
            # u = [vel, yaw_rate]
            ud = self.u + (np.matmul(np.random.randn(1, 2) / 4.0, R ** 0.5)).T
            # Se actualiza el estado de las particulas en funcion del motion model
            if WITH_EKF:
                px, p_cov = self.motion_model(px, ud, particles[i].p_cov)
                particles[i].p_cov = p_cov
            else:
                px = self.motion_model(px, ud)
            particles[i].x = px[0, 0]
            particles[i].y = px[1, 0]
            particles[i].yaw = px[2, 0]

        return particles

    def update_with_observation(self, particles, z):
        """
        Actualiza la posicion y covarianza de los landmarks de cada particula a partir de las nuevas observaciones.

        :param particles: Lista de todas las particulas con el vector de estado actualizado
        :param z: Observaciones detectadas [distancia, angulo, landmark_id]
        :return: Lista de particulas con el vector de estado y landmarks actualizadas
        """
        # Por cada particula
        for ip in range(N_PARTICLE):

            # Por cada landmark en z
            for iz in range(len(z[0, :])):
                landmark_id = int(z[2, iz])
                if abs(particles[ip].lm[landmark_id, 0]) == 0:
                    # Si la x es 0, se anade un landmark nuevo
                    particles[ip] = self.add_new_landmark(particles[ip], z[:, iz], Q)
                else:
                    # En cualquier otro caso, es un landmark ya conocido
                    xf = np.array(particles[ip].lm[landmark_id, :]).reshape(2, 1)
                    Pf = np.array(particles[ip].lmP[2 * landmark_id:2 * landmark_id + 2])
                    zp, Hv, Hf, Sf = self.compute_jacobians(particles[ip], xf, Pf, Q)
                    w = self.compute_weight(z[:, iz], zp, Sf)
                    # Se multiplica el peso por el guardado
                    particles[ip].w *= w
                    particles[ip] = self.update_landmark(particles[ip], z[:, iz], zp, xf, Pf, Hf)
                    particles[ip] = self.proposal_sampling(particles[ip], z[:, iz], Q)

        return particles

    def proposal_sampling(self, particle, z, Q_cov):
        '''
        Actualiza el estado de la particula calculando una covarianza y media, teniendo en cuenta la diferencia entre
        el landmark y la nueva observacion.

        :param particle: Particula con el vector de estado y landmarks actualizadas
        :param z: Una observacion detectada  [distancia, angulo, landmark_id]
        :param Q_cov: Matriz de covarianza de las observaciones de los conos
        :return: Particula con el estado actualizado
        '''
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
        # State
        x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
        P = particle.P
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        try:
            Sfi = np.linalg.inv(Sf)
            Pi = np.linalg.inv(P)
        except np.linalg.linalg.LinAlgError:
            print("singular")
            return 1.0

        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        particle.P = np.linalg.inv(np.matmul(np.matmul(Hv.T, Sfi), Hv) + Pi)  # proposal covariance
        x += np.matmul(np.matmul(np.matmul(particle.P, Hv.T), Sfi), dz)  # proposal mean

        particle.x = x[0, 0]
        particle.y = x[1, 0]
        particle.yaw = x[2, 0]

        return particle

    def add_new_landmark(self, particle, z, Q_cov):
        """
        Rellena la posicion y covarianza del nuevo landmark en cada particula

        :param particle: Particula con el vector de estado actualizado
        :param z: Una observacion [distancia, angulo, landmark_id]
        :param Q_cov: Matriz de covarianza de las observaciones de los conos
        :return: Particula con el vector de estado y covarianzas actualizadas
        """
        # Extraemos la distancia, el angulo y el landmark id
        r = z[0]
        b = z[1]
        lm_id = int(z[2])

        # Calculamos el seno y el coseno del angulo del coche mas el de la observacion
        s = math.sin(self.pi_2_pi(particle.yaw + b))
        c = math.cos(self.pi_2_pi(particle.yaw + b))

        # Se anaden la x y la y correspondientes a la lista de landmarks
        particle.lm[lm_id, 0] = particle.x + r * c
        particle.lm[lm_id, 1] = particle.y + r * s

        # Calculamos las distancias al coche de la landmark
        dx = r * c
        dy = r * s
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        # Jacobiano de la posicion de los landmarks
        Gz = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])
        particle.lmP[2 * lm_id:2 * lm_id + 2] = np.matmul(np.matmul(np.linalg.inv(Gz), Q_cov), np.linalg.inv(Gz.T))

        return particle

    def compute_weight(self, z, zp, Sf):
        """
        Calculo del peso de la particula (se multiplica luego por el peso anterior)

        :param z: Una observacion [distancia, angulo, landmark_id]
        :param zp: Distancia y angulo esperada del coche a un landmark globales [distancia, angulo]
        :param Sf: Covarianza de las medidas
        :return: El peso de la particula
        """
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
        Calcular los jacobianos de la particula y la distancia y angulo esperada del coche a un landmark globales.

        :param particle: Particula con el vector de estado actualizado
        :param xf: Posicion almacenada del landmark
        :param Pf: Covarianza almacenada del landmark
        :param Q_cov: Matriz de covarianza de las observaciones de los conos
        :return:
            zp: Distancia y angulo esperada del coche a un landmark globales [distancia, angulo]
            Hv: Jacobiano del vector de estado del coche
            Hf: Jacobiano de los landmarks
            Sf: Covarianza de las medidas
        """
        # Calculamos las distancias al coche de la landmark
        dx = xf[0, 0] - particle.x
        dy = xf[1, 0] - particle.y
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        # Calculamos el angulo del coche a la landmark en ejes globales
        zp = np.array([d, self.pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

        # Jacobiano del vector del coche
        Hv = np.array([[-dx / d, -dy / d, 0.0],
                       [dy / d2, -dx / d2, -1.0]])

        # Jacobiano de la posicion de los landmarks
        Hf = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])

        Sf = np.matmul(np.matmul(Hf, Pf), Hf.T) + Q_cov

        return zp, Hv, Hf, Sf

    def update_landmark(self, particle, z, zp, xf, Pf, Hf):
        """
        Actualiza la posicion y covarianza de un landmark guardado de acuerdo con las nuevas observaciones.

        :param z: Una observacion [distancia, angulo, landmark_id]
        :param zp: Distancia y angulo esperada del coche a un landmark globales [distancia, angulo]
        :param xf: Posicion almacenada del landmark
        :param Pf: Covarianza almacenada del landmark
        :param Hf: Jacobiano de los landmarks
        :return: Particula con el vector de estado y covarianzas actualizadas
        """
        # Obtiene el landmark id y la x y la y guardadas en lm
        lm_id = int(z[2])

        # La diferencia entre z y zp
        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        xf, Pf = self.update_kf_with_cholesky(xf, Pf, dz, Q, Hf)

        particle.lm[lm_id, :] = xf.T
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

        return particle

    def update_kf_with_cholesky(self, xf, Pf, v, Q_cov, Hf):
        '''
        Actualiza la posicion y covarianza del landmark segun el metodo de Cholesky.

        :param xf: Posicion almacenada del landmark
        :param Pf: Covarianza almacenada del landmark
        :param v: Diferencia entre z y zp
        :param Q_cov: Matriz de covarianza de las observaciones de los conos
        :param Hf: Jacobiano de los landmarks
        :return: Posicion y covarianza actualizada del landmark
        '''
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
        '''
        Compara la distribucion de los pesos y elimina las particulas con pesos mas bajos y, en su lugar, duplica los
        mas altos.

        :param particles: Particulas con los pesos actualizados
        :return: Particulas procesadas en caso de que la distribucion de pesos sea mala o, en caso contrario, las
                 particulas originales
        '''
        particles = self.normalize_weight(particles)

        pw = []
        for i in range(N_PARTICLE):
            pw.append(particles[i].w)

        # Array con todos los pesos de las particulas
        pw = np.array(pw)

        n_eff = 1.0 / (np.matmul(pw, pw.T))  # Effective particle number

        # Resampling cuando n_eff sea inferior al threshold
        if n_eff < NTH:  # resampling
            # Devuelve la suma acumulativa de los elementos del vector de pesos
            w_cum = np.cumsum(pw)
            # Vector tipo [0, 0.01, 0.02, ...., 0.99]
            base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
            # A la base se le suma un numero random dividido por 100
            resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE

            inds = []
            ind = 0
            # Se recorren todos los pesos y mientras que el resample sea mayor que el w_cum se introduce ese indice en los inds
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
        Normalizar los pesos para que la suma de todos sea 1.

        :param particles: Particulas con los pesos actualizados
        :return: Particulas con los pesos normalizados
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
        '''
        Calcula el estado del coche final teniendo en cuenta los pesos de cada particula.

        :param particles: Particulas actualizadas
        :return: Estado del coche final
        '''
        xEst = np.zeros((3, 1))

        particles = self.normalize_weight(particles)

        # Media de todas las posiciones con sus pesos
        for i in range(N_PARTICLE):
            xEst[0, 0] += particles[i].w * particles[i].x
            xEst[1, 0] += particles[i].w * particles[i].y
            xEst[2, 0] += particles[i].w * particles[i].yaw

        xEst[2, 0] = self.pi_2_pi(xEst[2, 0])

        return xEst

    def calc_final_lm_position(self, particles):
        '''
        Calcula la posicion final de los landmarks teniendo en cuenta los pesos de cada particula.

        :param particles: Particulas actualizadas
        :return: Posicion final de cada landmark
        '''
        landmarks_x = []
        landmarks_y = []

        particles = self.normalize_weight(particles)

        # Media de todas las posiciones con sus pesos
        for j in range(self.total_landmarks):
            xEst = 0.0
            yEst = 0.0
            for i in range(N_PARTICLE):
                xEst += particles[i].w * particles[i].lm[j, 0]
                yEst += particles[i].w * particles[i].lm[j, 1]
            landmarks_x.append(xEst)
            landmarks_y.append(yEst)

        return np.array(landmarks_x), np.array(landmarks_y)

    def get_best_particle(self, particles):
        ind = 0
        weight = 0.0

        # Media de todas las posiciones con sus pesos
        for i in range(N_PARTICLE):
            if particles[i].w > weight:
                ind = i

        return ind

    def get_best_position(self, particles):
        '''
        Calcula el estado del coche final cogiendo el de la particula con mejor peso.

        :param particles: Particulas actualizadas
        :return: Estado del coche final
        '''
        xEst = np.zeros((3, 1))

        particles = self.normalize_weight(particles)
        ind = self.get_best_particle(particles)

        xEst[0, 0] = particles[ind].x
        xEst[1, 0] = particles[ind].y
        xEst[2, 0] = particles[ind].yaw

        xEst[2, 0] = self.pi_2_pi(xEst[2, 0])

        return xEst

    def updateStep(self, gps_coord):
        '''
        Actualiza la posicion de cada particula con un EKF a partir del mensaje del gps.

        :param gps_coord: Coordenadas globales del coche segun el gps
        '''
        for particle in self.particles:
            # 3.1 Compute Kalman Gain
            R = self.var_gnss * np.eye(2)
            K = particle.p_cov.dot(self.h_jac.T.dot(np.linalg.inv(self.h_jac.dot(particle.p_cov.dot(self.h_jac.T)) + R)))

            # 3.2 Compute error state
            delta_x = K.dot(gps_coord - np.array([particle.x, particle.y]))

            # 3.3 Correct predicted state
            particle.x = particle.x + delta_x[0]
            particle.y = particle.y + delta_x[1]
            particle.yaw = particle.yaw + delta_x[2]

            # 3.4 Compute corrected covariance
            particle.p_cov = (np.eye(3) - K.dot(self.h_jac)).dot(particle.p_cov)

    def update_car_position(self, pose):
        '''
        Actualiza el estado del coche segun lo dado por el state estimation para luego comprobar que conos ya estan
        guardados y cuales son nuevos.

        :param pose: Posicion y orientacion del coche segun el mensaje de state estimation
        '''
        self.state_x = pose.position.x
        self.state_y = pose.position.y
        self.state_yaw = self.pi_2_pi(np.arctan2(2 * (pose.orientation.w * pose.orientation.z), 1 - 2 * (pose.orientation.z ** 2)))

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

    def local_to_global(self, x, y):
        '''
        Pasa de coordenadas locales a globales segun la posicion del coche obtenida del state estimation.

        :param x: Valor en el eje x en coordenadas locales
        :param y: Valor en el eje y en coordenadas locales
        :return: Coordenadas globales del punto
        '''
        xEst = self.get_best_position(self.particles)
        position = np.array([x, y])
        C_ns = np.array([[np.cos(xEst[2, 0]), -np.sin(xEst[2, 0])], [np.sin(xEst[2, 0]), np.cos(xEst[2, 0])]])
        rotated_position = C_ns.dot(position) + np.array([xEst[0, 0], xEst[1, 0]])

        return rotated_position[0], rotated_position[1]

    def rviz_update(self):
        '''
        Actualiza las variables que se mostraran en Rviz

        '''
        # Bucle para obtener el maximo peso de todas las particulas para mostrarlas con color diferente en Rviz
        max_weight = 0.0

        for particle in self.particles:
            if particle.w > max_weight:
                max_weight = particle.w

        # Funciones para Rviz
        self.marker_array(max_weight)
        self.marker_array_slam()
        self.publish_path()

    def marker_array(self, max_weight):
        '''
        MarkerArray para actualizar y mostrar en Rviz los estados de las particulas. En amarillo estan las particulas con
        pesos mas bajos y en verde los pesos mas altos.

        :param max_weight: Maximo peso de todas las particulas
        '''

        '''Verde constante en 255 y rojo en 0 para los mejores y 255 para los peores usando el peso mas alto y mas bajo'''
        # Publish it as a marker in rviz
        self.marker_ests.markers = []
        for i, particle in enumerate(self.particles):
            if max_weight == 0:
                value = 0
            else:
                value = 255 * (1 - particle.w / max_weight)

            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = particle.x
            point.y = particle.y
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
            marker_est.color.r, marker_est.color.g, marker_est.color.b = (value, 255, 0)
            marker_est.color.a = 0.5
            marker_est.scale.x, marker_est.scale.y, marker_est.scale.z = (0.05, 0.05, 0.4)
            self.marker_ests.markers.append(marker_est)

    def marker_array_slam(self):
        '''
        MarkerArray para actualizar y mostrar en Rviz la posicion de los conos que estan guardados.

        '''
        # Publish it as a marker in rviz
        self.slam_marker_ests.markers = []

        # Calcular las posiciones medias de los conos en las particulas
        landmarks_x, landmarks_y = self.calc_final_lm_position(self.particles)

        for i in range(len(landmarks_x)):
            marker_est = Marker()
            marker_est.header.frame_id = "map"
            marker_est.ns = "est_pose_" + str(i)
            marker_est.id = i
            marker_est.type = Marker.CYLINDER
            marker_est.action = Marker.ADD
            pose = Pose()
            point = Point()
            point.x = landmarks_x[i]
            point.y = landmarks_y[i]
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
