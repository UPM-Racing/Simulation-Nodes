#! /usr/bin/env python
import roslaunch
import rospy
from eufs_msgs.msg import CarState, CanState

class Master_node(object):
    def __init__(self):
        self.state_sub = rospy.Subscriber('/ros_can/state', CanState, self.callbackstate)

        self.state = 'OFF'

    def callbackstate(self, msg):
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
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/eufs_launcher.launch"])
    launch.start()
    master_node = Master_node()
    rospy.init_node('master_node', anonymous=True)
    rospy.loginfo("started")

    while master_node.state == 'OFF':
        continue

    if master_node.state == 'ACCELERATION':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'SKIDPAD':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/skidpad_sim.launch"])
        launch2.start()
    elif master_node.state == 'AUTOCROSS':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'TRACKDRIVE':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, ["/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/trackdrive_sim.launch"])
        launch2.start()
    elif master_node.state == 'AUTONOMOUS_DEMO':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'ADS_INSPECTION':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'ADS_EBS':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'DDT_INSPECTION_A':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'DDT_INSPECTION_B':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()
    elif master_node.state == 'MANUAL':
        launch2 = roslaunch.parent.ROSLaunchParent(uuid, [
            "/home/miguel/Documentos/EUFS/catkin_ws/src/eufs_sim-master/eufs_launcher/launch/nodos.launch"])
        launch2.start()

    #while (1):
    #    continue
    while not rospy.is_shutdown():
        continue