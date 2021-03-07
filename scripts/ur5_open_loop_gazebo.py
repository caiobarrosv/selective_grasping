#!/usr/bin/python
import time
import actionlib
import numpy as np
from copy import deepcopy
import sys
import re
import random

# ROS Msgs
from std_msgs.msg import Int16MultiArray, Float32MultiArray, String, Bool, Int8
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from controller_manager_msgs.srv import SwitchController

# ROS
import rosservice
import rospy
from tf import TransformListener, TransformBroadcaster
from tf.transformations import quaternion_from_euler

# Gazebo
from gazebo_msgs.msg import ContactsState, ContactState
from gazebo_msgs.srv import GetLinkState

# Robotiq
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg

# IK TRAJ
import common.IK_traj as IK_TRAJ

class TimeIt:
	def __init__(self, s):
		self.s = s
		self.t0 = None
		self.t1 = None
		self.print_output = False

	def __enter__(self):
		self.t0 = time.time()

	def __exit__(self, t, value, traceback):
		self.t1 = time.time()
		print('%s: %s' % (self.s, self.t1 - self.t0))

class ur5_grasp_project(object):
	def __init__(self, joint_values = None):
		rospy.init_node('command_GGCNN_ur5')
		self.joint_values_home = joint_values

		self.tf = TransformListener()

		# Used to change the controller
		self.controller_switch = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)

		# actionClient used to send joint positions
		self.client = actionlib.SimpleActionClient('pos_based_pos_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
		print("Waiting for server (pos_based_pos_traj_controller)...")
		self.client.wait_for_server()
		print("Connected to server (pos_based_pos_traj_controller)")
		
		#################
		# Gazebo related
		#################
		# Service proxy - Get model states in Gazebo
		self.get_model_coordinates = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
		
		# Publishers
		self.grasp_started = rospy.Publisher('grasp_started', Bool, queue_size=1) # Detection flag
		self.grasp_object_name = rospy.Publisher('grasp_object_name', String, queue_size=1) 
		self.grasp_object_position = rospy.Publisher('grasp_object_position', Pose, queue_size=1)
		
		# Subscribers
		# used to read joint values
		rospy.Subscriber('/joint_states', JointState, self.ur5_actual_position_callback, queue_size=1)
		rospy.Subscriber('/left_finger_bumper_vals', ContactsState, self.monitor_contacts_left_finger_callback) # ContactState
		rospy.Subscriber('/right_finger_bumper_vals', ContactsState, self.monitor_contacts_right_finger_callback) # ContactState
		rospy.sleep(1.0)
		
		# USED FOR COLLISION DETECTION
		self.finger_links = ['robotiq_85_right_finger_tip_link', 'robotiq_85_left_finger_tip_link']
		self.string = ""
		self.left_collision = False
		self.contactState_left = ContactState()
		self.right_collision = False
		self.contactState_right = ContactState()
		# Action clients
		self.client_gripper = actionlib.SimpleActionClient('gripper_controller_pos/follow_joint_trajectory', FollowJointTrajectoryAction)
		print("Waiting for server (gripper_controller_pos)...")
		self.client_gripper.wait_for_server()
		print("Connected to server (gripper_controller_pos)")
			
		#################
		# GGCNN Related 
		#################
		self.d = None
		self.ori_ = None
		self.grasp_cartesian_pose = None
		self.posCB = []
		self.ori = []
		self.gripper_angle_grasp = 0.0
		self.final_orientation = 0.0
		
		# These offsets are used only in Gazebo and should be calibrated for the real robot and camera
		self.GGCNN_offset_x = 0.0
		self.GGCNN_offset_y = 0.0
		self.GGCNN_offset_z = 0.020 # 0.019

		self.robotiq_joint_name = rospy.get_param("/robotiq_joint_name")

		# Topic published from GG-CNN Node
		rospy.Subscriber('ggcnn/out/command', Float32MultiArray, self.ggcnn_command_callback, queue_size=1)

		####################
		# Pipeline Related #
		####################
		self.classes = rospy.get_param("/classes")
		self.grasp_ready_flag = False
		self.detected_tags = []
		self.tags = ['tag_0_corrected', 'tag_1_corrected', 'tag_2_corrected', 'tag_3_corrected', 
				     'tag_4_corrected', 'tag_5_corrected', 'tag_6_corrected', 'tag_7_corrected']
		
		# These offset should also be calibrated for the real robot and camera
		self.tags_position_offset = [0.062, 0.0, 0.062]

		# Subscribers - Topics related to the new grasping pipeline
		rospy.Subscriber('flags/grasp_ready', Bool, self.grasp_ready_callback, queue_size=1) # Grasp flag
		rospy.Subscriber('flags/reposition_robot_flag', Bool, self.reposition_robot_callback, queue_size=1) # Reposition flag
		rospy.Subscriber('flags/detection_ready', Bool, self.detection_ready_callback ,queue_size=1) # Detection flag
		rospy.Subscriber('reposition_coord', Float32MultiArray, self.reposition_coord_callback, queue_size=1)
		rospy.Subscriber('/selective_grasping/tag_detections', Int16MultiArray, self.tags_callback, queue_size=1)

		# Publishers
		# Publish the required class so the other nodes such as gg-cnn generates the grasp to the selected part
		self.require_class = rospy.Publisher('pipeline/required_class', Int8, queue_size=1)
				
	def turn_velocity_controller_on(self):
		self.controller_switch(['joint_group_vel_controller'], ['pos_based_pos_traj_controller'], 1)

	def turn_position_controller_on(self):
		self.controller_switch(['pos_based_pos_traj_controller'], ['joint_group_vel_controller'], 1)

	def turn_gripper_velocity_controller_on(self):
		self.controller_switch(['gripper_controller_vel'], ['gripper_controller_pos'], 1)

	def turn_gripper_position_controller_on(self):
		self.controller_switch(['gripper_controller_pos'], ['gripper_controller_vel'], 1)

	def tags_callback(self, msg):
		self.detected_tags = msg.data

	def grasp_ready_callback(self, msg):
		self.grasp_ready_flag = msg.data
	
	def detection_ready_callback(self, msg):
		self.detection_ready_flag = msg.data
	
	def reposition_robot_callback(self, msg):
		self.reposition_robot_flag = msg.data
	
	def reposition_coord_callback(self, msg):
		self.reposition_coords = msg.data
	
	def monitor_contacts_left_finger_callback(self, msg):
		if msg.states:
			self.left_collision = True
			string = msg.states[0].collision1_name
			string_collision = re.findall(r'::(.+?)::',string)[0]
			# print("Left String_collision: ", string_collision)
			if string_collision in self.finger_links:
				string = msg.states[0].collision2_name
				# print("Left Real string (object): ", string)
				self.string = re.findall(r'::(.+?)::', string)[0]
				# print("Left before: ", self.string)
			else:
				self.string = string_collision
				# print("Left in else: ", string_collision)
		else:
			self.left_collision = False

	def monitor_contacts_right_finger_callback(self, msg):
		if msg.states:
			self.right_collision = True
			string = msg.states[0].collision1_name
			string_collision = re.findall(r'::(.+?)::',string)[0]
			# print("Right String_collision: ", string_collision)
			if string_collision in self.finger_links:
				string = msg.states[0].collision2_name
				# print("Right Real string (object): ", string)
				self.string = re.findall(r'::(.+?)::',string)[0]
				# print("Right before: ", self.string)
			else:
				self.string = string_collision
				# print("Right in else: ", self.string)
		else:
			self.right_collision = False

	def ur5_actual_position_callback(self, joint_values_from_ur5):
		"""Get UR5 joint angles
		
		The joint states published by /joint_states of the UR5 robot are in wrong order.
		/joint_states topic normally publishes the joint in the following order:
		[elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
		But the correct order of the joints that must be sent to the robot is:
		['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
		
		Arguments:
			joint_values_from_ur5 {list} -- Actual angles of the UR5 Robot
		"""
		self.th3, self.robotic, self.th2, self.th1, self.th4, self.th5, self.th6 = joint_values_from_ur5.position
		self.actual_position = [self.th1, self.th2, self.th3, self.th4, self.th5, self.th6]

	def ggcnn_command_callback(self, msg):
		"""
		GGCNN Command Subscriber Callback
		"""
		self.tf.waitForTransform("base_link", "object_detected", rospy.Time(0), rospy.Duration(4.0)) # rospy.Time.now()
		object_pose, object_ori = self.tf.lookupTransform("base_link", "object_detected", rospy.Time(0))
		self.d = list(msg.data)
		object_pose[0] += self.GGCNN_offset_x
		object_pose[1] += self.GGCNN_offset_y
		object_pose[2] += self.GGCNN_offset_z
		
		self.posCB = object_pose

		self.ori = self.d[3]
		br = TransformBroadcaster()
		br.sendTransform((object_pose[0], 
							object_pose[1],
							object_pose[2]), 
							quaternion_from_euler(0.0, 0.0, self.ori),
							rospy.Time.now(),
							"object_link",
							"base_link")
		
	def get_link_position_picking(self):
		link_name = self.string
		model_coordinates = self.get_model_coordinates(self.string, 'wrist_3_link')
		self.model_pose_picking = model_coordinates.link_state.pose
		self.grasp_object_position.publish(Pose(self.model_pose_picking.position, self.model_pose_picking.orientation))
		print("OK 2")

	def all_close(self, goal, tolerance=0.00005):
		"""Wait until goal is reached in configuration space
		
		This method check if the robot reached goal position since wait_for_result seems to be broken

		
		Arguments:
			goal {[list]} -- Goal in configuration space (joint values)
		
		Keyword Arguments:
			tolerance {number} -- Minimum error allowed to consider the trajectory completed (default: {0.00005})
		"""

		error = np.sum([(self.actual_position[i] - goal[i])**2 for i in range(6)])
		# print("> Waiting for the trajectory to finish...")
		while not rospy.is_shutdown() and error > tolerance:
			error = np.sum([(self.actual_position[i] - goal[i])**2 for i in range(6)])
		# if error < tolerance:
			# print("> Trajectory Suceeded!\n") # whithin the tolerance specified
		# else:
			# rospy.logerr("> Trajectory aborted.")
	
	def traj_planner(self, grasp_position, grasp_step='move', way_points_number=10, movement='slow'):
		"""Quintic Trajectory Planner
	
		Publish a trajectory to UR5 using quintic splines. 

		Arguments:
			grasp_position {[float]} -- Grasp position [x, y, z]

		Keyword Arguments:
			grasp_step {str} -- Set UR5 movement type (default: {'move'})
			way_points_number {number} -- Number of points considered in trajectory (default: {10})
			movement {str} -- Movement speed (default: {'slow'})
		"""
		if grasp_step == 'pregrasp':
			self.grasp_cartesian_pose = deepcopy(self.posCB)
			self.ori_ = deepcopy(self.ori)

		grasp_cartesian_pose = self.grasp_cartesian_pose
		posCB = self.posCB
		ori = self.ori_
		actual_position = self.actual_position
		d = self.d

		status, goal, joint_pos = IK_TRAJ.traj_planner(grasp_position, grasp_step, way_points_number, movement, 
													   grasp_cartesian_pose, ori, d, actual_position)
		if status:
			self.client.send_goal(goal)
			self.all_close(joint_pos)
		else:
			print("Could not retrieve any IK solution")

	def gripper_send_position_goal(self, position=0.3, velocity=0.4, action='move'):
		"""Send position goal to the gripper
		
		Keyword Arguments:
			position {float} -- Gripper angle (default: {0.3})
			velocity {float} -- Gripper velocity profile (default: {0.4})
			action {str} -- Gripper movement (default: {'move'})
		"""

		self.turn_gripper_position_controller_on()
		duration = 0.2
		if action == 'pre_grasp_angle':
			gripper_angle_grasp = deepcopy(self.d[-2])
			max_distance = 0.085
			angular_coeff = 0.11
			K = 1 # lower numbers = higher angles
			angle = (max_distance - gripper_angle_grasp) / angular_coeff * K
			position = angle
			velocity = 0.4
		elif action == 'pick':
			position = 0.75
			velocity = 0.05
			duration = 8.0

		goal = FollowJointTrajectoryGoal()
		goal.trajectory = JointTrajectory()
		goal.trajectory.joint_names = self.robotiq_joint_name
		goal.trajectory.points.append(JointTrajectoryPoint(positions=[position],
														   velocities=[velocity],
														   accelerations=[0.0],
														   time_from_start=rospy.Duration(duration)))
		self.client_gripper.send_goal(goal)
		if action == 'pick':
			while not rospy.is_shutdown() and not self.left_collision and not self.right_collision:
				pass
			self.client_gripper.cancel_goal()
		
	def move_on_shutdown(self):
		self.client.cancel_goal()
		self.client_gripper.cancel_goal()
		print("Shutting down node...")
	
	def grasp_main(self, point_init_home, depth_shot_point):
		self.classes =  ["bar_clamp", "gear_box", "vase", "part_1", "part_3", "nozzle"]
		recognition_fail = 0
		grasping_fail = 0
		total_attempts = 0
		random_classes = [i for i in range(len(self.classes))]

		bin_location = [[-0.65, -0.1, 0.2],
						[-0.65, 0.1, 0.2]]
		garbage_location = [-0.4, -0.30, 0.10]
		
		raw_input('=== Press enter to start the grasping process')	

		while not rospy.is_shutdown():
			with TimeIt('Grasping time'):
				# The grasp is performed randomly and the chosen object is published to
				# the detection node
				if not len(random_classes):
					print("\n> IMPORTANT! There is no object remaining in the object's picking list. Resetting the objects list...\n")
					random_classes = [i for i in range(len(self.classes))]
				
				random_object_id = random.sample(random_classes, 1)[0]
				random_classes.pop(random_classes.index(random_object_id))
				print('> Object to pick: ', self.classes[random_object_id])
					
				# Publish the required class ID so the other nodes such as gg-cnn
				# generates the grasp to the selected part
				self.require_class.publish(random_object_id)

				# raw_input("==== Press enter to start the grasping process!")
				rospy.sleep(1.0)
				if self.detection_ready_flag:
					# Check if the robot needs to be repositioned to reach the object. This flag is published
					# By the detection node. The offset coordinates (self.reposition_coords) are also published 
					# by the detection node
					if self.reposition_robot_flag:
						# print('> Repositioning robot...')
						self.tf.waitForTransform("base_link", "grasping_link", rospy.Time.now(), rospy.Duration(4.0))
						eef_pose, _ = self.tf.lookupTransform("base_link", "grasping_link", rospy.Time(0))
						
						corrected_position = [eef_pose[0] - self.reposition_coords[1],
											eef_pose[1] + self.reposition_coords[0],
											eef_pose[2]]
		
						self.traj_planner(corrected_position, movement='fast')
						# raw_input("==== Press enter when the trajectory finishes!")
						rospy.sleep(2.0)
								
					# Check if:
					# - The run_ggcn is ready (through grasp_ready_flag);
					# - The detection node is ready (through detection_ready_flag);
					# - The robot does not need to be repositioned (through reposition_robot_flag)
					if self.grasp_ready_flag and self.detection_ready_flag and not self.reposition_robot_flag:
						if not self.detection_ready_flag:
							recognition_fail += 1
						
						if not self.grasp_ready_flag:
							grasping_fail += 1

						# raw_input("==== Press enter to move to the pre grasp position")
						# print('> Moving to the grasping position... \n')
						self.traj_planner([], 'pregrasp', movement='fast')					
											
						# raw_input("==== Press enter to move the gripper")
						# # It closes the gripper a little before approaching the object
						# # to prevent colliding with other objects
						# self.gripper_send_position_goal(action='pre_grasp_angle')
						
						# raw_input("==== Press enter to move to the grasp position")
						# # Moving the robot to pick the object - BE CAREFULL!
						# self.traj_planner([], 'grasp', movement='slow')
						
						# # raw_input('==== Press enter to pick the object')
						# print("> Picking object...")				
						# self.gripper_send_position_goal(action='pick')
						# # Send commands to the publish_objects_path node to move the objects in gazebo
						# # It attaches the object to the gripper
						# self.get_link_position_picking()
						# self.grasp_started.publish(True)
						# self.grasp_object_name.publish(self.string)
						# raw_input("==== Continue")
											
						# # After a collision is detected, the arm will start the picking action
						# # Different locations were adopted because the camera field of view is not big enough
						# # to detect all the april tags in the environment
						# # raw_input("==== Press enter to move the object to the bin")
						# if random_object_id < 5:
						# 	self.traj_planner(bin_location[0], movement='fast')
						# else:
						# 	self.traj_planner(bin_location[1], movement='fast')

						# rospy.sleep(2.0) # waiting for the tag detections

						# selected_tag_status = self.detected_tags[random_object_id]
						# print('selected tag: ', self.tags[random_object_id])
						# # Check if the tag corresponding to the object ID was identified in order to
						# # move the object there
						# if selected_tag_status:
						# 	self.tf.waitForTransform("base_link", self.tags[random_object_id], rospy.Time(0), rospy.Duration(2.0)) # rospy.Time.now()
						# 	ptFinal, oriFinal = self.tf.lookupTransform("base_link", self.tags[random_object_id], rospy.Time(0))
						# 	ptFinal = [ptFinal[i] + self.tags_position_offset[i] for i in range(3)]
						# 	# raw_input("==== Move to the tag")
						# 	pt_inter = deepcopy(ptFinal)
						# 	# pt_inter adds 0.1m to the Z coordinate of ptFinal in order to
						# 	# approach the box before leaving the object there
						# 	pt_inter[-1] += 0.1
						# 	self.traj_planner(pt_inter, movement='fast')
						# 	self.traj_planner(ptFinal, movement='fast')
						# else:
						# 	# In this case, the camera identifies some other tag but not the tag corresponding to the object
						# 	print('> Could not find the box tag. Placing the object anywhere \n')
						# 	# raw_input('=== Pres enter to proceed')
						# 	self.traj_planner(garbage_location, movement='fast')

						# print("> Placing object...")					
						# # After the bin location is reached, the robot will place the object and move back
						# # to the initial position
						# self.gripper_send_position_goal(0.3)
						# self.grasp_started.publish(False)
						
						# print("> Moving back to home position...\n")
						self.traj_planner(point_init_home, movement='fast')
						self.traj_planner(depth_shot_point, movement='fast')
						# print('> Grasping finished \n')
					else:
						# print('> The requested object could not be grasped! \n')
						self.traj_planner(depth_shot_point, movement='fast')

				else:
					recognition_fail += 1
					# print('> The requested object was not detected! \n')
				
				total_attempts += 1
				print("Grasping attempts: ", total_attempts)
				print("Grasping fail: ", grasping_fail)
				print("Recognition fails: ", recognition_fail)
				
def main():
	ur5_vel = ur5_grasp_project()
	point_init_home = [-0.37, 0.11, 0.15]
	joint_values_home = IK_TRAJ.get_ik(point_init_home)
	ur5_vel.joint_values_home = joint_values_home

	# Send the robot to the custom HOME position
	raw_input("==== Press enter to 'home' the robot!")
	rospy.on_shutdown(ur5_vel.move_on_shutdown)
	ur5_vel.traj_planner(point_init_home, movement='fast')

	depth_shot_point = [-0.37, 0.11, 0.05]

	# Remove all objects from the scene and press enter
	raw_input("==== Press enter to move the robot to the 'depth cam shot' position!")
	ur5_vel.traj_planner(depth_shot_point, movement='fast')

	print("> Starting the gripper in Gazebo! Please wait...")
	ur5_vel.gripper_send_position_goal(0.4)
		
	print('> Gripper started \n')	
	ur5_vel.grasp_main(point_init_home, depth_shot_point)

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		print("Program interrupted before completion")
