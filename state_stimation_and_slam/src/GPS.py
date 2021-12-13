#! /usr/bin/env python
import rospy
import math

class GPS():
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