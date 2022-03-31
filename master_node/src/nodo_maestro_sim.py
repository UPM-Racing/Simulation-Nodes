#! /usr/bin/env python
from sys import path
import roslaunch
import rospy
from eufs_msgs.msg import CarState, CanState
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties

import os
import re

MAX_UPDATE_RATE = 100

class Master_node(object):
    def __init__(self):
        self.state_sub = rospy.Subscriber('/ros_can/state', CanState, self.callbackstate)

        self.gazebo_initialized = False

        self.state = 'OFF'

    def callbackstate(self, msg):
        if not self.gazebo_initialized:
            self.gazebo_initialized = True

        if msg.ami_state == 10:
            self.state = 'OFF'
        elif msg.ami_state == 11:
            self.state = 'ACCELERATION'
        elif msg.ami_state == 12:
            self.state = 'SKIDPAD'
        elif msg.ami_state == 13:
            self.state = 'AUTOCROSS'
        elif msg.ami_state == 14:
            self.state = 'TRACKDRIVE'
        elif msg.ami_state == 15:
            self.state = 'AUTONOMOUS_DEMO'
        elif msg.ami_state == 16:
            self.state = 'ADS_INSPECTION'
        elif msg.ami_state == 17:
            self.state = 'ADS_EBS'
        elif msg.ami_state == 18:
            self.state = 'DDT_INSPECTION_A'
        elif msg.ami_state == 19:
            self.state = 'DDT_INSPECTION_B'
        elif msg.ami_state == 20:
            self.state = 'MANUAL'


if __name__ == "__main__":
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    # Relative path obtention
    current_path = os.getcwd()

    launch_path = None
    for root, dirs, files in os.walk(current_path):
        if re.search("eufs_launcher/launch$", root):
            launch_path = root

    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/eufs_launcher.launch"])
    launch.start()
    master_node = Master_node()
    rospy.init_node('master_node', anonymous=True)
    rospy.loginfo("started")

    # Wait until gazebo initializes
    while not master_node.gazebo_initialized:
        rospy.sleep(0.1)

    # Get the current physics properties and change the max_update_rate
    try:
        gazebo_get_physics_properties = rospy.ServiceProxy("gazebo/get_physics_properties", GetPhysicsProperties)
        gazebo_set_physics_properties = rospy.ServiceProxy("gazebo/set_physics_properties", SetPhysicsProperties)
        properties = gazebo_get_physics_properties()

        response = gazebo_set_physics_properties(properties.time_step, MAX_UPDATE_RATE, properties.gravity, properties.ode_config)
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))


    while master_node.state == 'OFF':
        rospy.sleep(0.1)

    if master_node.state == 'ACCELERATION':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/acceleration_sim.launch"])
        launch2.start()
    elif master_node.state == 'SKIDPAD':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/skidpad_sim.launch"])
        launch2.start()
    elif master_node.state == 'AUTOCROSS':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/autocross_sim.launch"])
        launch2.start()
    elif master_node.state == 'TRACKDRIVE':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/trackdrive_sim.launch"])
        launch2.start()
    elif master_node.state == 'AUTONOMOUS_DEMO':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()
    elif master_node.state == 'ADS_INSPECTION':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()
    elif master_node.state == 'ADS_EBS':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()
    elif master_node.state == 'DDT_INSPECTION_A':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()
    elif master_node.state == 'DDT_INSPECTION_B':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()
    elif master_node.state == 'MANUAL':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [launch_path + "/nodos.launch"])
        launch2.start()

    try:
        rospy.spin()
    finally:
        launch2.shutdown()
        launch.shutdown()
