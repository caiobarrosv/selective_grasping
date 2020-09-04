#!/usr/bin/env python
import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel, GetModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

class spawn_objects():
	def __init__(self):
		rospy.init_node('spawn_model')

		rospack = rospkg.RosPack()
		self.Home = rospack.get_path('selective_grasping')

		self.Spawning1 = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)
		rospy.wait_for_service("gazebo/spawn_sdf_model")
		model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

		# The spawn need some time to wait the UR5 to show in Gazebo
		object_coordinates = model_coordinates("robot", "")
		self.z_position = object_coordinates.pose.position.z
		self.y_position = object_coordinates.pose.position.y
		self.x_position = object_coordinates.pose.position.x

	def spawning(self, model_name, path, ptFinal, oriFinal):
		obj_path = self.Home + path
		
		with open(obj_path) as f:
			product_xml = f.read()

		print("Spawning model: {}".format(model_name))

		# X and Y positions are somewhat in an incorrect order in Gazebo
		item_pose = Pose(Point(x=self.y_position + ptFinal[0], y=self.x_position + ptFinal[1], z=self.z_position + ptFinal[2]),
						 Quaternion(oriFinal[0], oriFinal[1], oriFinal[2], oriFinal[3]))
		
		self.Spawning1(model_name, product_xml, "", item_pose, "world")

if __name__ == '__main__':
	spawn_obj = spawn_objects()


	object_path = '/models/b0/model.sdf'
	ptFinal = [0.33, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b0', object_path, ptFinal, oriFinal)

	object_path = '/models/b1/model.sdf'
	ptFinal = [0.25, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b1', object_path, ptFinal, oriFinal)

	object_path = '/models/b2/model.sdf'
	ptFinal = [0.17, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b2', object_path, ptFinal, oriFinal)

	object_path = '/models/b3/model.sdf'
	ptFinal = [0.09, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b3', object_path, ptFinal, oriFinal)

	object_path = '/models/b4/model.sdf'
	ptFinal = [0.01, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b4', object_path, ptFinal, oriFinal)

	object_path = '/models/b5/model.sdf'
	ptFinal = [-0.07, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b5', object_path, ptFinal, oriFinal)

	object_path = '/models/b6/model.sdf'
	ptFinal = [-0.15, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b6', object_path, ptFinal, oriFinal)

	object_path = '/models/b7/model.sdf'
	ptFinal = [-0.23, -0.74, 0.027]
	oriFinal = quaternion_from_euler(0.0, 0.0, 1.57)
	spawn_obj.spawning('b7', object_path, ptFinal, oriFinal)