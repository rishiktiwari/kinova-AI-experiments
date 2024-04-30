import time
import math
import PIL
import torch
import numpy as np
import cv2 as cv

# clipseg stuff
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# custom class
from commander import Commander



class PrimitiveActions:
	REQUIRED_ARGS = {
		'pick': 1,
		'pick_place': 2,
		'ignore': 0
	}

	def __init__(self) -> None:
		VLM_CACHE_DIR = "/Users/rishiktiwari/AI/VLMs_test/hf_models"
		VLM_NAME = "CIDAS/clipseg-rd64-refined"

		self.TK_DEF_LABEL = "How may I assist?"
		self.tkLabelCurrentText = self.TK_DEF_LABEL

		self.cartPose = (0.3, 0.3, 0.3, 0.0) # to track past/current pose
		self.cmndr = Commander()
		self.cmndr.initCommandLink()
		self.cmndr.sendCommand(self.cartPose)

		self.endScript = False
		self.isEnacting = False
		self.isDetecting = False
		self.grasped = False
		self.isPlacing = False
		self.gripper = 0.0
		self.noObjFrames = 0 # num of frames with no detected objs
		self.bbox_samples = []
		self.gripperEst_samples = []

		self.device = 'cpu'
		self.llmIntel = None
		self.isFresh_rgbd = False
		self.rgbd_frame = (None, None) # rgb, depth image array

		self.clipseg_processor = CLIPSegProcessor.from_pretrained(VLM_NAME, cache_dir=VLM_CACHE_DIR)
		self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(VLM_NAME, cache_dir=VLM_CACHE_DIR).to('cpu')

		self.vlm_ann_rgbframe = np.zeros(shape=(100, 100, 3), dtype=np.uint8)
		self.vlm_heatmap = np.zeros(shape=(100, 100), dtype=np.uint8)
		# self.vlm_ann_depthframe = np.zeros(shape=(100, 100, 3), dtype=np.uint8)



	def resetFlagsAndParams(self, evt = None) -> None:
		self.isEnacting = False
		self.grasped = False
		self.isPlacing = False
		self.llmIntel = None
		self.tkLabelCurrentText = self.TK_DEF_LABEL
		self.noObjFrames = 0
		self.vlm_ann_rgbframe = np.zeros(shape=(100, 100, 3), dtype=np.uint8)
		self.vlm_heatmap = np.zeros(shape=(100, 100), dtype=np.uint8)
		self.bbox_samples = []
		self.gripperEst_samples = []
		self.gripper = 0.0
		self.cartPose = (0.3, 0.3, 0.3, 0.0)
		self.cmndr.sendCommand(self.cartPose) # home the robot
		return None



	def normaliseDepthForPreview(self, rawDepthImage, reshapeOneChannel = False):
		depthFrameNormalised = np.nan_to_num(rawDepthImage, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
		depthFrameNormalised = (depthFrameNormalised * 255).round().astype(np.uint8)

		if reshapeOneChannel:
			# reshape from WxH to WxHxC
			depthFrameNormalised = np.repeat(depthFrameNormalised[:, :, np.newaxis], 3, axis=2)
		
		return depthFrameNormalised



	def _takeObjToHome(self, pickCoords: tuple) -> None:
		# moves the already grasped obj to home, takes it back to pick location after N time and releases
		if not self.grasped:
			print("nothing grasped to take home")
			return None

		print("taking obj to home")
		self.cmndr.sendCommand((0.3, 0.3, 0.3, pickCoords[3])) # move to home pose
		print("holding for some time")
		time.sleep(float(10.0))
		self.isPlacing = True
		self.tkLabelCurrentText = 'Placing object...'
		print("taking obj to pickup location")
		self.cmndr.sendCommand(pickCoords) # move to pick pose
		time.sleep(2.0)
		print("releasing obj at pick location")
		self.cmndr.sendCommand((pickCoords[0], pickCoords[1], pickCoords[2], 0.0)) # release obj
		time.sleep(2.0)
		self.resetFlagsAndParams()
		return None

	

	def _applyAndGetVisionDetail(self, rgbImage, rawDepthImage, objectName: str) -> dict|None:			
		if not self.isFresh_rgbd or type(rgbImage) == None or type(rawDepthImage) == None:
			return None
		
		GRIPPER_LENGTH_CM = 22
		# DEPTH_OFFSET = -2.5 # real arm
		DEPTH_OFFSET = 3 # sim
		self.isFresh_rgbd = False
		colour = (0, 255, 0) # green
		rgbFrameImg = PIL.Image.fromarray(rgbImage[:,:,::-1], mode='RGB')
		detail = {
			'contours': [],
			'bbox': (0,0,0,0),			# ox, oy, w, h 
			'bbox_center': (0,0),		# cx, cy
			'extremes': (0,0,0,0),		# left, top, right, bottom
			'extremes_center': (0,0),	# ex, ey
			'gripper_center': (0,0,0), 	# mx, my, tolPx
			'grasp_estimate': 0.0,		# gripper position estimate from vision, 0 is full open, 1 is full close
			'obj_mid_depth': 0.0		# scaled and corrected depth at automatically decided cx,cy or ex,ey or obj_seg_mean
		}

		# -- PREDICTION START --
		start_time = time.perf_counter()
		vlm_inputs = self.clipseg_processor(
			text=[objectName],
			images=[rgbFrameImg],
			padding="max_length",
			return_tensors="pt"
		).to('cpu') # does not support mps
		with torch.no_grad():
			vlm_outputs = self.clipseg_model(**vlm_inputs)

		end_time = time.perf_counter()
		print("VLM inference time: %.2fms" % ((end_time-start_time)*1000))
		# -- PREDICTION END --
		
		# -- PREDICTION PROCESSING --
		preds = vlm_outputs.logits.unsqueeze(1)
		
		i = 0 # TODO: Add multiple obj handling. Assuming only 1 obj for now and ignoring rest.
		norm_map = (np.array(torch.sigmoid(preds[i][0]))*255).astype(dtype=np.uint8) # sig activation -> scale -> round to int
		norm_map = cv.resize(norm_map, dsize=rgbImage.shape[:2]) # def size is 352x352, so resize to match rgb image
		
		# extract contours
		(_, thresh) = cv.threshold(norm_map, 178, 255, cv.THRESH_BINARY)
		(detail['contours'], _) = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)

		if(len(detail['contours']) == 0):
			print('no obj contours found')
			self.isDetecting = False
			self.noObjFrames += 1
			if self.noObjFrames == 6:
				print('--- no objs, terminating action ---')
				self.resetFlagsAndParams()
			return None

		cnt = detail['contours'][0]
		# img_moments = cv.moments(cnt)
		# cx = int(img_moments['m10']/img_moments['m00'])
		# cy = int(img_moments['m01']/img_moments['m00'])

		# calculate bbox from contours
		(ox, oy, ow, oh) = cv.boundingRect(cnt)

		# find rect, calculate moving avg, and draw obj bbox
		self.bbox_samples.append((ox, oy, ow, oh))
		self.bbox_samples = self.bbox_samples[-5:] # recent N values
		samples_count = len(self.bbox_samples)     # starts with 0, maxes at window size
		# ox = sum(i[0] for i in self.bbox_samples)//samples_count
		# oy = sum(i[1] for i in self.bbox_samples)//samples_count
		ow = sum(i[2] for i in self.bbox_samples)//samples_count
		oh = sum(i[3] for i in self.bbox_samples)//samples_count
		cx = max(0, min(ox+(ow//2), rgbImage.shape[0]-1)) # clamp value 0 <= cx < image.shape[n]
		cy = max(0, min(oy+(oh//2), rgbImage.shape[1]-1)) # clamp value 0 <= cy < image.shape[n]

		detail['bbox'] = (ox, oy, ow, oh) # width and height are filtered with moving average
		detail['bbox_center'] = (cx, cy)

		self.vlm_heatmap = norm_map # both images should be set consequently for correct display
		self.vlm_ann_rgbframe = rgbImage.copy()
		cv.putText(self.vlm_ann_rgbframe, objectName, (ox, oy-5), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=colour)
		cv.rectangle(self.vlm_ann_rgbframe, (ox, oy),(ox+ow, oy+oh), color=colour, thickness=2)
		cv.circle(self.vlm_ann_rgbframe, (cx, cy), radius=5, color=colour, thickness=-1)
		# self.vlm_ann_depthframe = self.normaliseDepthForPreview(rawDepthImage, True)
		# cv.rectangle(self.vlm_ann_depthframe, (ox, oy),(ox+ow, oy+oh), color=colour, thickness=1)

		# EXTREME POINTS
		leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
		rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
		topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
		bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
		detail['extremes'] = (leftmost, topmost, rightmost, bottommost)
		detail['extremes_center'] = ((leftmost[0]+rightmost[0])//2, (topmost[1]+bottommost[1])//2)

		# diff and scale extreme x-axis obj points, take moving avg of N window size
		self.gripperEst_samples.append(round((rightmost[0]-leftmost[0])/120, 2))
		self.gripperEst_samples = self.gripperEst_samples[-3:] # recent N values
		samples_count = len(self.gripperEst_samples)
		detail['grasp_estimate'] = min(1.0, max(0.5, round(sum(self.gripperEst_samples)/samples_count, 2))) # clamp value 0.5 <= grasp_estimate <= 1.0

		cv.circle(self.vlm_ann_rgbframe, leftmost, radius=3, color=(255,255,255), thickness=-1)
		cv.circle(self.vlm_ann_rgbframe, rightmost, radius=3, color=(0,0,255), thickness=-1)
		cv.circle(self.vlm_ann_rgbframe, topmost, radius=3, color=(0,255,255), thickness=-1)
		cv.circle(self.vlm_ann_rgbframe, bottommost, radius=3, color=(0,255,255), thickness=-1)
		cv.circle(self.vlm_ann_rgbframe, detail['extremes_center'], radius=5, color=(255,0,0), thickness=-1) # obj mid point determined from extreme seg points

		# automatic object depth estimation
		detail['obj_mid_depth'] = rawDepthImage[cx, cy] * 100 - (GRIPPER_LENGTH_CM + DEPTH_OFFSET) # TODO: handle automatically which value to choose
		# fallback to mean if mid is nan
		if (math.isnan(detail['obj_mid_depth'])):
			print('>> TODO using mean depth <<')
			# TODO: implement mean distance based on seg mask
			# mean_region = rawDepthImage[cy-bh//2:cy+bh//2, cx-bw//2:cx+bw//2]
			# mid_depth = np.nanmean(mean_region) * 100 - (GRIPPER_LENGTH_CM - 2.5)
   
		# -- ARM STUFF --
		(h, w) = rgbImage.shape[:2]
		mx = (h//4)*3   # shift required mid point to lower half near the gripper
		my = w//2
		tolPx = 20 if detail['obj_mid_depth'] > 12 else 30 # tolerance region
		detail['gripper_center'] = (mx, my, tolPx)

		# draw ref lines
		cv.line(self.vlm_ann_rgbframe, (0, mx), (w, mx), colour, 1) # horz
		cv.line(self.vlm_ann_rgbframe, (my, 0), (my, h), colour, 1) # vert
		cv.rectangle(self.vlm_ann_rgbframe, (my-tolPx, mx-tolPx), (my+tolPx, mx+tolPx), color=(0,255,255), thickness=1) # tolerance region
		return detail



	def pick(self, object_name: str) -> None:
		if self.grasped:
			print("not picking, already grasped")
			return None
		
		print("initiating pick obj: %s" % object_name)
		self.tkLabelCurrentText = 'Picking...'
		self.isEnacting = True
		self.isDetecting = True
		colour = (0, 255, 0) # green
		pickCoords = None

		while (not self.endScript and not self.grasped and self.isEnacting):
			rgbImage = self.rgbd_frame[0]
			rawDepthImage = self.rgbd_frame[1]

			vision = self._applyAndGetVisionDetail(rgbImage, rawDepthImage, object_name)
			if vision == None:
				continue

			(cx, cy) = vision['bbox_center']
			(mx, my, tolPx) = vision['gripper_center']
			mid_depth = vision['obj_mid_depth']
			est_gripper = vision['grasp_estimate']

			stepSize = 0.02 if mid_depth > 12 else 0.01

			print('> Object: %s, Mid: %.1fcm' % (object_name, mid_depth))

			stepx = 0.0
			stepy = 0.0
			stepz = 0.0

			xAligned = False
			yAligned = False

			if(cx > my+tolPx):
				stepx = stepSize
			elif(cx < my-tolPx):
				stepx = -stepSize
			else:
				xAligned = True
			
			if(cy > mx+tolPx):
				stepy = -stepSize
			elif(cy < mx-tolPx):
				stepy = stepSize
			else:
				yAligned = True

			if (xAligned and yAligned): # vertically aligned
				if (mid_depth >= 1.5):
					print('> --- decr z')
					stepz = -stepSize
				elif (mid_depth < 1.5):
					print('> --- grasp: %.1f%%' % (est_gripper*100,))
					self.gripper = est_gripper
					self.grasped = True

			step = np.add(self.cartPose, (stepx, stepy, stepz, 0.0))
			step[3] = self.gripper # gripper is set, not added
			self.cartPose = tuple(step)
			if self.grasped:
				pickCoords = tuple(step)

			self.cmndr.sendCommand(self.cartPose)
			self.cmndr.awaitCommandFeedback() # awaits until command complete feedback is received

		# takes obj to home, holds for some time and brings back to pick location
		self._takeObjToHome(pickCoords)

		self.isEnacting = False
		self.isDetecting = False
		return True
	


	def pick_place(self, object_name: str, target_name: str) -> None:
		print("TODO: pick %s and place at: %s" % (object_name, target_name))
		time.sleep(3)
		self.resetFlagsAndParams()
		print('action complete')

	def ignore(self, *args) -> None:
		print("Do nothing")
