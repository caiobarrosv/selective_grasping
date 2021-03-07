import numpy as np

import rospy
from tf.transformations import quaternion_from_euler
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Inverse kinematics
from trac_ik_python.trac_ik import IK

ur5_joint_names = rospy.get_param("/ur5_joint_names")

def get_ik(position):
	"""Get the inverse kinematics 
	
	Get the inverse kinematics of the UR5 robot using track_IK package giving a desired intial position
	
	Arguments:
		position {list} -- A position representing x, y and z
	Returns:
		sol {list} -- Joint angles or None if track_ik is not able to find a valid solution
	"""
	camera_support_angle_offset = 0.0
	
	q = quaternion_from_euler(0.0, -3.14 + camera_support_angle_offset, 0.0)
	# Joint order:
	# ('shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'grasping_link')            
	ik_solver = IK("base_link", "grasping_link", solve_type="Distance")
	sol = ik_solver.get_ik([0.2201039360819781, -1.573845095552878, -1.521853400505349, -1.6151347051274518, 1.5704492904506875, 0.0], 
			position[0], position[1], position[2], q[0], q[1], q[2], q[3])
	
	if sol is not None:
		sol = list(sol)
		sol[-1] = 0.0
	else:
		print("IK didn't return any solution")
		
	return sol

def __build_goal_message_ur5():
	goal = FollowJointTrajectoryGoal()
	goal.trajectory = JointTrajectory()
	goal.trajectory.joint_names = ur5_joint_names
	goal.goal_tolerance.append(JointTolerance('joint_tolerance', 0.1, 0.1, 0))
	goal.goal_time_tolerance = rospy.Duration(5,0)
	return goal

def traj_planner(grasp_position, grasp_step, way_points_number, movement, grasp_cartesian_pose, ori, d, actual_position):
	"""Quintic Trajectory Planner
	
	Publish a trajectory to UR5 using quintic splines. 
	
	Arguments:
		grasp_position {[float]} -- Grasp position [x, y, z]
		grasp_step {str} -- Set UR5 movement type
		way_points_number {number} -- Number of points considered in trajectory
		movement {str} -- Movement speed
	"""
	offset_from_object = 0.05
	if grasp_step == 'pregrasp':
		grasp_cartesian_pose[-1] += offset_from_object
		joint_pos = get_ik(grasp_cartesian_pose)
		if joint_pos is not None:
			joint_pos[-1] = ori
			# final_orientation = ori
			# gripper_angle_grasp = d[-2]
	elif grasp_step == 'grasp':
		grasp_cartesian_pose[-1] -= offset_from_object
		joint_pos = get_ik(grasp_cartesian_pose)
		if joint_pos is not None:
			joint_pos[-1] = ori
	elif grasp_step == 'move':
		joint_pos = get_ik(grasp_position)
		if joint_pos is not None:
			joint_pos[-1] = 0.0
	
	if joint_pos is not None:
		if movement=='slow':
			final_traj_duration = 500.0 # total iteractions
		elif movement=='fast':
			final_traj_duration = 300.0
		v0 = a0 = vf = af = 0
		t0 = 5.0
		tf = (t0 + final_traj_duration) / way_points_number # tf by way point
		t = tf / 10 # for each movement
		ta = tf / 10 # to complete each movement
		a = [0.0]*6
		pos_points, vel_points, acc_points = [0.0]*6, [0.0]*6, [0.0]*6
		
		goal = __build_goal_message_ur5()
		for i in range(6):
			q0 = actual_position[i]
			qf = joint_pos[i]
			b = np.array([q0,v0,a0,qf,vf,af]).transpose()
			m = np.array([[1, t0, t0**2,   t0**3,    t0**4,    t0**5],
						[0,  1,  2*t0, 3*t0**2,  4*t0**3,  5*t0**4],
						[0,  0,     2,    6*t0, 12*t0**2, 20*t0**3],
						[1, tf, tf**2,   tf**3,    tf**4,    tf**5],
						[0,  1,  2*tf, 3*tf**2,  4*tf**3,  5*tf**4],
						[0,  0,     2,    6*tf, 12*tf**2, 20*tf**3]])
			a[i] = np.linalg.inv(m).dot(b)
		for i in range(way_points_number):
			for j in range(6):
				pos_points[j] =   a[j][0] +   a[j][1]*t +    a[j][2]*t**2 +    a[j][3]*t**3 +   a[j][4]*t**4 + a[j][5]*t**5
				vel_points[j] =   a[j][1] + 2*a[j][2]*t +  3*a[j][3]*t**2 +  4*a[j][4]*t**3 + 5*a[j][5]*t**4
				acc_points[j] = 2*a[j][2] + 6*a[j][3]*t + 12*a[j][4]*t**2 + 20*a[j][5]*t**3
			goal.trajectory.points.append(JointTrajectoryPoint(positions=pos_points,
															velocities=vel_points,
															accelerations=acc_points,
															time_from_start=rospy.Duration(t))) #default 0.1*i + 5
			t += ta
		return True, goal, joint_pos
	else:
		print('Could not find a IK solution')
		return False, None, None