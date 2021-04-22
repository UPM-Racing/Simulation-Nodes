#! /usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Path


class ErrorMeasurementClass(object):
    def __init__(self):
        # self.state_sub = rospy.Subscriber('/path_pub', Path, self.state_callback)
        self.state_sub = rospy.Subscriber('/slam_path_pub', Path, self.state_callback)
        self.ground_truth_sub = rospy.Subscriber('/ground_path_pub', Path, self.ground_truth_callback)

        self.state_path = Path()
        self.ground_truth_path = Path()

    def state_callback(self, msg):
        self.state_path = msg

    def ground_truth_callback(self, msg):
        self.ground_truth_path = msg

    def error_measurement(self):
        ground_index = 0
        state_index = 0
        distancias = []
        while(ground_index < len(self.ground_truth_path.poses) and state_index < len(self.state_path.poses)):
            ground_pose = self.ground_truth_path.poses[ground_index]
            time_sec = ground_pose.header.stamp.secs
            time_nsec = float(ground_pose.header.stamp.nsecs) / (10.0 ** 9)
            ground_time = time_sec + time_nsec

            state_pose = self.state_path.poses[state_index]
            time_sec = state_pose.header.stamp.secs
            time_nsec = float(state_pose.header.stamp.nsecs) / (10.0 ** 9)
            state_time = time_sec + time_nsec

            if abs(ground_time - state_time) < 0.01:
                ground_position = np.array([ground_pose.pose.position.x, ground_pose.pose.position.y])
                state_position = np.array([state_pose.pose.position.x, state_pose.pose.position.y])
                distancias.append(np.linalg.norm(ground_position - state_position))
                state_index += 1
                ground_index += 1
            elif ground_time > state_time:
                state_index += 1
            else:
                ground_index += 1

        rms = np.sqrt(np.mean(np.square(np.array(distancias))))

        if rms > 0:
            print(rms)

if __name__ == '__main__':
    rospy.init_node('error_measurement_node', anonymous=True)
    error_measurement = ErrorMeasurementClass()
    rate = rospy.Rate(0.5)  # Hz

    while not rospy.is_shutdown():
        error_measurement.error_measurement()
        rate.sleep()