#!/usr/bin/env python
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import numpy as np
import mxnet as mx
import gluoncv as gcv
from gluoncv.model_zoo import get_model
import gluoncv.data.transforms.image as timage
import gluoncv.data.transforms.bbox as tbbox
import cv2
import time
import argparse

# ROS related
import rospy
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Bool, Int8 # String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospkg

def parse_args():
	parser = argparse.ArgumentParser(description='GG-CNN Node')
	parser.add_argument('--gazebo', action='store_true', help='Set properties based on the Gazebo environment')
	args = parser.parse_args()
	return args

args = parse_args()

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
        print('%s: %s ms' % (self.s, (self.t1 - self.t0)*1000))

class Detector(object):
	def __init__(self, param, model_name='ssd300', ctx='gpu', filter_threshold=0.5, nms_thresh=0.5):
		self.filter_threshold = filter_threshold
		
		###############
		# ROS Related #
		###############

		rospy.init_node('obj_detection', anonymous=True)

		self.horizontal_FOV = rospy.get_param("/GGCNN/FOV")
		self.vertical_FOV = rospy.get_param("/GGCNN/vertical_FOV")
		
		# Publish the image with the bounding boxes to ROS
		self.img_pub = rospy.Publisher('img/bounding_box', Image, queue_size=1)
		# Publish the bounding boxes coordinates
		self.bb_points_array = rospy.Publisher('bb_points_array', Int32MultiArray, queue_size=10)
		self.label_array = rospy.Publisher('label_array', Int32MultiArray, queue_size=10)
		self.detection_ready = rospy.Publisher('flags/detection_ready', Bool, queue_size=1) # Detection flag
		self.reposition_robot_flag = rospy.Publisher('flags/reposition_robot_flag', Bool, queue_size=1) # Detection flag
		self.reposition_coord = rospy.Publisher('reposition_coord', Float32MultiArray, queue_size=10)

		# Transform from ROS image to OpenCV image type
		self.bridge = CvBridge()
		
		# Subscribe to the image published in Gazebo
		rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)
		rospy.Subscriber("pipeline/required_class", Int8, self.required_pipeline_class_callback, queue_size=10)
		if not args.gazebo:
			camera_topic = rospy.get_param("/GGCNN/camera_topic_realsense")
		else:
			camera_topic = rospy.get_param("/GGCNN/camera_topic")	
		rospy.Subscriber(camera_topic, Image, self.get_depth_callback, queue_size=10)
				
		###################
		# GluonCV Related #
		###################

		# Choose the default processing unit
		if ctx == 'cpu':
			self.ctx = mx.cpu()
		elif ctx == 'gpu':
			self.ctx = mx.gpu(0)
		else:
			raise ValueError('Invalid context.')

		# Load GluonCV parameters
		self.model_names_param = rospy.get_param("/model_names")
		self.width = self.model_names_param[model_name]['width']
		self.height = self.model_names_param[model_name]['height']
		print("Model name: ", model_name)
		print("Width: ", self.width)
		print("Height: ", self.height)
		print("Filter threshold: {} %".format(filter_threshold*100))

		# TODO
		# network = self.model_names_param[model_name]['network']
		# if network == 'ssd':
			# self.transform = transforms.SSDDefaultValTransform(self.width, self.height)

		self.classes = rospy.get_param("/classes")
		print('classes: ', self.classes)
		# Get the pre-trained model
		net = get_model(model_name, pretrained=False, ctx=self.ctx)
		# net.set_nms(nms_thresh=0.5, nms_topk=2)
		net.hybridize(static_alloc=True, static_shape=True)
		net.initialize(force_reinit=True, ctx=self.ctx)
		net.reset_class(classes=self.classes)
		
		# Load the parameter stored in the ROS package folder
		rospack=rospkg.RosPack()
		path = rospack.get_path("selective_grasping")
		param_path = path + "/params/" + param
		net.load_parameters(param_path, ctx=self.ctx)
		
		self.net = net

		# Used to transform the image as done in the training
		self.mean = (0.485, 0.456, 0.406)
		self.std = (0.229, 0.224, 0.225)
		self.depth_img_height = 480
		self.depth_img_width = 640

		self.pipeline_required_class = False
		self.receive_bb_status = False # Indicates if the 

	def filter_predictions(self, bounding_boxes, scores, class_IDs):
		threshold = self.filter_threshold
		idx = scores.squeeze().asnumpy() > threshold
		fscores = scores.squeeze().asnumpy()[idx]
		fids = class_IDs.squeeze().asnumpy()[idx]
		fbboxes = bounding_boxes.squeeze().asnumpy()[idx]
		return fbboxes, fscores, fids

	def get_depth_callback(self, depth_image):
		self.depth_image = self.bridge.imgmsg_to_cv2(depth_image)

	def image_callback(self, color_msg):
		color_img = self.bridge.imgmsg_to_cv2(color_msg)
		self.color_img = color_img
	
	def required_pipeline_class_callback(self, msg):
		self.pipeline_required_class = msg.data
		print('class: ', self.classes[self.pipeline_required_class])

	def resize_bounding_boxes(self, bounding_box):
		"""
		Transforms the bounding box generated in the RGB image to the Depth image
		Obs: necessary when the RBG and the Depth image are not aligned (gazebo case)
		"""
		center_calibrated_point = np.array([312, 240])
		K = 0.2
		offset = 10

		bbox_list = []

		for bbox in bounding_box:
			center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
			dist = [int(center[0] - center_calibrated_point[0]), int(center[1] - center_calibrated_point[1])]
			final_distance = [int(dist[0]*K), int(dist[1]*K)]
			start_point = (bbox[0] + final_distance[0] - offset, bbox[1] + final_distance[1] - offset)
			end_point = (bbox[2] + final_distance[0] + offset, bbox[3] + final_distance[1] + offset)

			# The new adjusted bounding box coordinates 
			bb_coordinates = [start_point[0], start_point[1], end_point[0], end_point[1]]
			bbox_list.append(bb_coordinates)
		return bbox_list

	def network_inference(self):
		# a = cv2.waitKey(0) # close window when ESC is pressed
		# while a is not 27:
		color_img = self.color_img
		depth_image = self.depth_image

		depth_height_res, depth_width_res = depth_image.shape

		# It is to correct the image size to fit a perfect square
		# color_img = np.zeros((640, 640, 3)).astype('uint8')
		# color_img[0:479] = color_img_raw[0:479]
		# color_img = color_img.astype('uint8')

		# Image pre-processing
		frame = mx.nd.array(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)).astype('uint8')
		frame = timage.imresize(frame, self.width, self.height, 1)
		frame_tensor = mx.nd.image.to_tensor(frame)
		frame_tensor = mx.nd.image.normalize(frame_tensor, mean=self.mean, std=self.std)
		
		# with TimeIt('Obj detection time'):
		# Run frame through network
		class_IDs, scores, bounding_boxes = self.net(frame_tensor.expand_dims(axis=0).as_in_context(self.ctx))
		
		# Filter bounding boxes by their scores
		fbounding_boxes, fscores, fclass_IDs = self.filter_predictions(bounding_boxes, scores, class_IDs)

		# we need to resize the bounding box back to the original resolution (640, 480) (width, height)
		resized_bbox = tbbox.resize(fbounding_boxes, (self.width, self.height), (self.depth_img_width, self.depth_img_height))
		img = timage.imresize(frame, self.depth_img_width, self.depth_img_height, 1)

		# check if the bounding box is inside the 300x300 area of the GG-CNN grasping area
		GGCNN_area = [190, 0, 480, 300]
		GGCNN_area_center = [320, 150] # width, height

		img_2 = img.asnumpy()
		img = cv2.rectangle(img_2, (GGCNN_area[0], GGCNN_area[1]), (GGCNN_area[2], GGCNN_area[3]), (255, 0, 0), 1)
				
		bbox_list, fscores_list, fclass_IDs_list = [], [], [] # bounding boxes of the chosen class
		bbox_list_backup = bbox_list
		# If any object is found
		if fclass_IDs.size > 0:
			# If the request object is found
			if self.pipeline_required_class in fclass_IDs:
				print('found obj')
				# we need to find all ocurrences of the class identified to consider
				# situation where we have false positives as well
				chosen_class_index = [i for i, x in enumerate(fclass_IDs) if x == self.pipeline_required_class]
				for class_index in chosen_class_index:
					bbox_list.append(resized_bbox[class_index])
					fscores_list.append(fscores[class_index])
					fclass_IDs_list.append(fclass_IDs[class_index])
					
				max_score = max(fscores_list)
				largest_score_bb_index = [i for i, x in enumerate(fscores_list) if x == max_score]
				
				bbox_list = [bbox_list[largest_score_bb_index[0]]]
				fscores_list = [fscores_list[largest_score_bb_index[0]]]
				fclass_IDs_list = [fclass_IDs_list[largest_score_bb_index[0]]]

				bbox_list_backup = bbox_list
				bbox_list = self.resize_bounding_boxes(bbox_list)
				self.labels = fclass_IDs_list
				self.bboxes = bbox_list

				for index, bbox in enumerate(bbox_list):
					if bbox[0] > GGCNN_area[0] and bbox[1] > GGCNN_area[1] and bbox[2] < GGCNN_area[2] and \
						bbox[3] < GGCNN_area[3]:
						print('obj inside ggcnn_area')

						self.receive_bb_status = True

						# Set the flag detection_ready
						self.detection_ready.publish(True)
						self.reposition_robot_flag.publish(False)
					else:
						print('obj outside ggcnn_area')

						bbox_center_point_x = (bbox[2] - bbox[0])/2 + bbox[0] # width
						bbox_center_point_y = (bbox[3] - bbox[1])/2 + bbox[1] # height

						dist_x = bbox_center_point_x - GGCNN_area_center[0] # width
						dist_y = GGCNN_area_center[1] - bbox_center_point_y # height

						dist_x_dir = dist_x/abs(dist_x)
						dist_y_dir = dist_y/abs(dist_y)

						ggcnn_center_area = depth_image[GGCNN_area_center[1], GGCNN_area_center[0]]
						
						self.horizontal_FOV = 52
						self.vertical_FOV = 60
												
						largura_2 = 2.0 * ggcnn_center_area * np.tan(self.horizontal_FOV * abs(dist_x) / depth_width_res / 2.0 / 180.0 * np.pi) / 1000 * dist_x_dir
						altura_2 = 2.0 * ggcnn_center_area * np.tan(self.vertical_FOV * abs(dist_y) / depth_height_res / 2.0 / 180.0 * np.pi) / 1000 * dist_y_dir

						reposition_points = Float32MultiArray()
						reposition_points.data = [largura_2, altura_2]
						self.reposition_coord.publish(reposition_points)

						self.detection_ready.publish(True)
						self.reposition_robot_flag.publish(True)
			else:
				print('The object ({}) was not found'.format(self.classes[self.pipeline_required_class]))
				self.detection_ready.publish(False)
				self.reposition_robot_flag.publish(False)
		else:
			print('No objects (including the requested one ({})) were found'.format(self.classes[self.pipeline_required_class]))
			self.detection_ready.publish(False)
			self.reposition_robot_flag.publish(False)

		# bbox_list = np.array(bbox_list)
		bbox_list = np.array(bbox_list_backup)
		fscores_list = np.array(fscores_list)
		fclass_IDs_list = np.array(fclass_IDs_list)

		img = gcv.utils.viz.cv_plot_bbox(img, bbox_list, fscores_list, fclass_IDs_list, class_names=self.net.classes)		
		depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
		depth_image = depth_image.astype('uint8')
		img = img.astype('uint8')
		added_image = cv2.addWeighted(depth_image, 0.7, img, 0.8, 0)
			
		self.img_pub.publish(CvBridge().cv2_to_imgmsg(img, 'bgr8'))
				
	def detect_main(self):
		"""
		Publish the label_array - ID of the identified part
		"""
		color_img = self.color_img

		points_to_send = Int32MultiArray()
		labels_to_send = Int32MultiArray()
	
		rate = rospy.Rate(4)
		while not rospy.is_shutdown():
			with TimeIt('detection_process'):
				self.network_inference()
				if self.receive_bb_status:
					labels = self.labels
					bboxes = self.bboxes
					size = len(bboxes)
					if size != 0:
						points_to_send_list = []
						labels_to_send_list = []
						for label, bbox in zip(labels, bboxes):
							labels_to_send_list.append(int(label))
							points_to_send_list.append(int(bbox[0]))
							points_to_send_list.append(int(bbox[1]))
							points_to_send_list.append(int(bbox[2]))
							points_to_send_list.append(int(bbox[3]))

						points_to_send.data = points_to_send_list # assign the array with the value you want to send
						labels_to_send.data = labels_to_send_list
						self.bb_points_array.publish(points_to_send)
						self.label_array.publish(labels_to_send)
						points_to_send.data = []
						labels_to_send.data = []
					self.receive_bb_status = False
				rate.sleep()
		
def main():
	# TODO: You just need to pass the param name inside the log folder (checkpoints folder configured in config.json)
	# params = 'ssd_512_resnet50_v1_coco_best_epoch_0047_map_0.9656.params'
	# params = 'ssd_512_resnet50_v1_voc_best_epoch_0038_map_0.9652.params'
	params = 'ssd_300_vgg16_atrous_voc_best_epoch_0028_map_0.9392.params'
	params = 'ssd_512_vgg16_atrous_coco_best_epoch_0079_map_0.9535'

	obj_detect = Detector(params, 
						  model_name='ssd_300_vgg16_atrous_voc', 
						  ctx='gpu', 
						  filter_threshold=0.8, 
						  nms_thresh=0.5)

	obj_detect.detect_main()

if __name__ == "__main__":
	main()