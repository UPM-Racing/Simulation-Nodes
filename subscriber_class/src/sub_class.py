#! /usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray
import math
from eufs_msgs.msg import WheelSpeedsStamped

class Gps():

  def __init__(self):
    self.a = 6378137
    self.b = 6356752.3142
    self.f = (self.a - self.b) / self.a
    self.e_sq = self.f * (2 - self.f)

    self.LONGITUD_0 = 0
    self.LATITUD_0 = 0
    self.ALTURA_0 = 0

  def geodetic_to_ecef(self, lat, lon, h):
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

    return xEast, yNorth, zUp

  def geodetic_to_enu(self, lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = self.geodetic_to_ecef(lat, lon, h)

    return self.ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

  def gps_loop(self, timestamp, latitude, longitude, altitude):
    if self.LONGITUD_0 == 0:
      self.LONGITUD_0 = longitude
      self.LATITUD_0 = latitude
      self.ALTURA_0 = altitude

    gps_coordinates = self.geodetic_to_enu(latitude, longitude, altitude, self.LATITUD_0, self.LONGITUD_0, self.ALTURA_0)

    gps_values = [timestamp, gps_coordinates[0], gps_coordinates[1], gps_coordinates[2]]

    return gps_values

class SensorClass(object):
  def __init__(self):
    self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_odom_callback)
    self.imu_pub = rospy.Publisher('/imu_pub', Imu, queue_size=1)
    self.imu = Imu()

    self.odom_sub = rospy.Subscriber('/ros_can/wheel_speeds', WheelSpeedsStamped, self.odom_callback)
    self.odometry = WheelSpeedsStamped()
    self.odom_pub = rospy.Publisher('/odometry_pub', WheelSpeedsStamped, queue_size=1)

    self.gps = Gps()
    #self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)
    #self.gps_pub = rospy.Publisher('/gps_pub', Float64MultiArray, queue_size=1)

  def gps_callback(self, msg):
    latitude = msg.latitude
    longitude = msg.longitude
    altitude = msg.altitude
    time_sec = msg.header.stamp.secs
    time_nsec = msg.header.stamp.nsecs/(10.0**9)
    timestamp = time_sec + time_nsec

    gps = Float64MultiArray()
    gps.data = self.gps.gps_loop(timestamp, latitude, longitude, altitude)
    self.gps_pub.publish(gps)

  def imu_callback(self, msg):
    imudata_angular = msg.angular_velocity
    imudata_linear = msg.linear_acceleration
    time_sec = msg.header.stamp.secs
    time_nsec = msg.header.stamp.nsecs/(10.0**9)
    timestamp = time_sec + time_nsec

    imu = Float64MultiArray()
    imu.data = [timestamp, imudata_angular.x, imudata_angular.y, imudata_angular.z, imudata_linear.x, imudata_linear.y, imudata_linear.z]
    self.imu_pub.publish(imu)

  def imu_odom_callback(self, msg):
    self.imu = msg

  def odom_callback(self, msg):
    self.odometry = msg

if __name__ == '__main__':
  rospy.init_node('sensor_node', anonymous=True)
  sensor_class = SensorClass()
  # rospy.spin()
  rate = rospy.Rate(70)  # Hz

  while not rospy.is_shutdown():
    sensor_class.odom_pub.publish(sensor_class.odometry)
    sensor_class.imu_pub.publish(sensor_class.imu)
    rate.sleep()