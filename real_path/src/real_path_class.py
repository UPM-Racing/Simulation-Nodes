#! /usr/bin/env python
import rospy
from eufs_msgs.msg import CarState
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class RealPathClass(object):
  def __init__(self):
    self.ground_truth_sub = rospy.Subscriber('/ground_truth/state', CarState, self.sub_callback)

    self.real_path = Path()
    self.ground_path_pub = rospy.Publisher('/ground_path_pub', Path, queue_size=1)

    #self.ground_pub = rospy.Publisher('/ground_pose_pub', PoseStamped, queue_size=1)

  def sub_callback(self, msg):
    self.real_path.header = msg.header
    self.real_path.header.frame_id = "map"
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose

    self.real_path.poses.append(pose)
    #self.ground_path_pub.publish(self.real_path)
    #self.ground_pub.publish(pose)


if __name__ == '__main__':
  rospy.init_node('ground_path_node', anonymous=True)
  real_path_class = RealPathClass()
  #rospy.spin()
  rate = rospy.Rate(5)  # Hz

  while not rospy.is_shutdown():
    real_path_class.ground_path_pub.publish(real_path_class.real_path)
    rate.sleep()