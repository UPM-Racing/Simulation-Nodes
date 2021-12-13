#! /usr/bin/env python
import rospy
import math
import numpy as np
import csv

from eufs_msgs.msg import ConeArrayWithCovariance
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import NavSatFix
from ackermann_msgs.msg import AckermannDriveStamped
from GPS import GPS

# import Ekf_Slam as slam
import Slam2 as slam
FASTSLAM = True

class Slam_Class(object):
    def __init__(self):
        # Inicializacion de variables
        self.slam = slam.SLAM()
        if FASTSLAM:
            if slam.WITH_EKF:
                self.gps = GPS()

        ''' Topicos de ROS '''
        # Subscriber de la entrada de observaciones de conos
        self.cones_sub = rospy.Subscriber('/ground_truth/cones', ConeArrayWithCovariance, self.cones_callback)

        # Subscriber de las entradas del modelo cinematico
        self.control_sub = rospy.Subscriber('/cmd_vel_out', AckermannDriveStamped, self.control_callback)
        self.gps_vel_sub = rospy.Subscriber('/gps_velocity', Vector3Stamped, self.gps_vel_callback)

        # Subscribers para la correccion de la posicion
        if FASTSLAM:
            if slam.WITH_EKF:
                self.gps_sub = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)

        # Publishers de SLAM
        self.path_pub = rospy.Publisher('/slam_path_pub', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/slam_pose_pub', PoseStamped, queue_size=1)
        self.slam_marker_array_pub = rospy.Publisher('/slam_marker_array_pub', MarkerArray, queue_size=1)
        if FASTSLAM:
            self.marker_array_pub = rospy.Publisher('/particles_marker_array_pub', MarkerArray, queue_size=1)

    def cones_callback(self, msg):
        time_sec = msg.header.stamp.secs
        time_nsec = float(msg.header.stamp.nsecs) / (10.0 ** 9)
        timestamp = time_sec + time_nsec
        cones = msg.yellow_cones + msg.blue_cones + msg.big_orange_cones
        self.slam.bucle_principal(cones, timestamp)

    def control_callback(self, msg):
        angle = msg.drive.steering_angle
        acceleration = msg.drive.acceleration

        self.slam.update_car_steer(angle)

    def gps_vel_callback(self, msg):
        velocity = [-msg.vector.y + 1 * 10 ** -12, msg.vector.x]
        velocity_mean = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
        self.slam.update_car_gps_vel(velocity_mean)

    def gps_callback(self, msg):
        latitude = msg.latitude
        longitude = msg.longitude
        altitude = msg.altitude
        x, y = self.gps.gps_to_local(latitude, longitude, altitude)
        gps_values = np.array([x, y])
        self.slam.updateStep(gps_values)


if __name__ == '__main__':
    rospy.init_node('slam_node', anonymous=True)
    slam_class = Slam_Class()
    rate = rospy.Rate(10) # Frecuencia de los publishers (Hz)
    '''with open('../catkin_ws/results/Slam1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
        writer.writerow(['SLAM 1 Period'])

    with open('../catkin_ws/results/cones_position_slam1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', lineterminator='\n', )
        writer.writerow(['Cone number', 'X', 'Y'])'''

    while not rospy.is_shutdown():
        if FASTSLAM:
            slam_class.marker_array_pub.publish(slam_class.slam.marker_ests)
        slam_class.slam_marker_array_pub.publish(slam_class.slam.slam_marker_ests)
        slam_class.path_pub.publish(slam_class.slam.path)
        slam_class.pose_pub.publish(slam_class.slam.pose)
        rate.sleep()