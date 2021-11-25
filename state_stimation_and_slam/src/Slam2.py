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

from Particle import Particle
from Slam1 import SLAM as FastSLAM1

WITH_EKF = True

Q = np.diag([0.3, np.deg2rad(5.0)]) ** 2
R = np.diag([0.5, np.deg2rad(30.0)]) ** 2

N_PARTICLE = 50  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling

class SLAM(FastSLAM1):
    def __init__(self):
        super(SLAM, self).__init__()

        # Constantes para Fast SLAM 2.0
        self.THRESHOLD = 1.5
        self.var_gnss = 0.25
        self.var_model = 0.1

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