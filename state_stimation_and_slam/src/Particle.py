#! /usr/bin/env python

import numpy as np

class Particle():
    def __init__(self, n_landmark, weight, WITH_EKF):
        # Estado del coche y landmarks en la particula
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lm = np.zeros((n_landmark, 2)) # Posicion x-y de cada landmark

        # Variables para los calculos
        self.w = weight # Peso de la particula
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