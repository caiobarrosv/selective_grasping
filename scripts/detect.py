#!/usr/bin/env python
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon
from gluoncv.data.transforms.presets import ssd, rcnn
from gluoncv.model_zoo import get_model
import gluoncv.data.transforms.image as timage
import gluoncv.data.transforms.bbox as tbbox
import cv2
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import time
from gluoncv.utils.bbox import bbox_iou 

# ROS related
import rospy
from std_msgs.msg import Int32MultiArray # String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import rospkg

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
		# Publish the image with the bounding boxes to ROS
		self.img_pub = rospy.Publisher('img/bouding_box', Image, queue_size=1)
		# Publish the bounding boxes coordinates
		self.arraypub = rospy.Publisher('bb_points_array', Int32MultiArray, queue_size=10)
		# Subscribe to the image published in Gazebo
		rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=10)		
		# Transform from ROS image to OpenCV image type
		self.bridge = CvBridge()

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
		path = rospack.get_path("ssggcnn_ur5_grasping")
		param_path = path + "/params/" + param
		net.load_parameters(param_path, ctx=self.ctx)
		
		self.net = net

		self.mean = (0.485, 0.456, 0.406)
		self.std = (0.229, 0.224, 0.225)

	def filter_predictions(self, bounding_boxes, scores, class_IDs):
		threshold = self.filter_threshold
		idx = scores.squeeze().asnumpy() > threshold
		fscores = scores.squeeze().asnumpy()[idx]
		fids = class_IDs.squeeze().asnumpy()[idx]
		fbboxes = bounding_boxes.squeeze().asnumpy()[idx]
		return fbboxes, fscores, fids

	def image_callback(self, color_msg):
		color_img = self.bridge.imgmsg_to_cv2(color_msg)
		# height_res, width_res, _ = color_img.shape
		# color_img = color_img[0 : self.crop_size, 
					# (width_res - self.crop_size)//2 : (width_res - self.crop_size)//2 + self.crop_size]
		self.color_img = color_img

	def network_inference(self):
		# a = cv2.waitKey(0) # close window when ESC is pressed
		# while a is not 27:
		color_img = self.color_img

		# Image pre-processing
		frame = mx.nd.array(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)).astype('uint8')
		frame = timage.imresize(frame, self.width, self.height, 1)
		frame_tensor = mx.nd.image.to_tensor(frame)
		frame_tensor = mx.nd.image.normalize(frame_tensor, mean=self.mean, std=self.std)
		
		with TimeIt('Processing time'):
			# Run frame through network
			class_IDs, scores, bounding_boxes = self.net(frame_tensor.expand_dims(axis=0).as_in_context(self.ctx))
		
		# Filter bounding boxes by their scores
		fbounding_boxes, fscores, fclass_IDs = self.filter_predictions(bounding_boxes, scores, class_IDs)
		
		if fclass_IDs.size > 0:
			img = gcv.utils.viz.cv_plot_bbox(frame, fbounding_boxes, fscores, fclass_IDs, class_names=self.net.classes)
		
		# Uncomment this to plot also using OpenCV - Remember to use cv2.waitKey()
		# gcv.utils.viz.cv_plot_image(img)
		
		self.img_pub.publish(CvBridge().cv2_to_imgmsg(img, 'bgr8'))
		# a = cv2.waitKey(1) # close window when ESC is pressed 
		self.labels = fclass_IDs
		self.scores = fscores
		self.bboxes = fbounding_boxes

	def detect_main(self):
		color_img = self.color_img

		points_to_send = Int32MultiArray()	
		rate = rospy.Rate(10) # 10hz 
		while not rospy.is_shutdown():
			self.network_inference()
			labels = self.labels
			scores = self.scores
			bboxes = self.bboxes
			size = len(bboxes)
			if size != 0:
				points_to_send_list = []
				for bbox in bboxes:
					point1= Point()
					point2= Point()
					point1.x=int(bbox[0])
					point1.y=int(bbox[1])
					point2.x=int(bbox[2])
					point2.y=int(bbox[3])

					points_to_send_list.append(point1.x)
					points_to_send_list.append(point1.y)
					points_to_send_list.append(point2.x)
					points_to_send_list.append(point2.y)

			
			points_to_send.data = points_to_send_list # assign the array with the value you want to send
			print(points_to_send.data)
			self.arraypub.publish(points_to_send)
			points_to_send.data = []
			rate.sleep()
		
def main():
	# TODO: You just need to pass the param name inside the log folder (checkpoints folder configured in config.json)
	params = 'ssd_512_resnet50_v1_voc_best_epoch_0017_map_0.9534.params'

	obj_detect = Detector(params, 
						  model_name='ssd_512_resnet50_v1_voc', 
						  ctx='gpu', 
						  filter_threshold=0.5, 
						  nms_thresh=0.5)

	obj_detect.detect_main()

if __name__ == "__main__":
	main()