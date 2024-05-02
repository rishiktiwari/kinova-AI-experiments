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

	GRIPPER_LENGTH_CM = 22
	# DEPTH_OFFSET = -2.5 # real arm
	DEPTH_OFFSET = 3 # sim
	CONTOUR_THRESHOLD = 178

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
		self.vlm_ann_depthframe = np.zeros(shape=(100, 100, 3), dtype=np.uint8)

		# self.vlm_depth_compute_frame = np.zeros(shape=(100, 100, 3), dtype=np.uint8)



	def resetFlagsAndParams(self, evt = None) -> None:
		self.isEnacting = False
		self.isDetecting = False
		self.isPlacing = False
		self.grasped = False
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
		depthFrameNormalised = np.nan_to_num(rawDepthImage, copy=True, nan=0.5, posinf=1.0, neginf=0.0)
		depthFrameNormalised = (depthFrameNormalised * 255).round().astype(np.uint8)

		if reshapeOneChannel:
			# reshape from WxH to WxHxC
			depthFrameNormalised = np.repeat(depthFrameNormalised[:, :, np.newaxis], 3, axis=2)
		
		return depthFrameNormalised
	
	

	def _rawDepthToCm(self, rawDepthValue: float) -> float:
		return float(rawDepthValue * 100 - (self.GRIPPER_LENGTH_CM + self.DEPTH_OFFSET))



	def _applyAndGetVisionDetail(self, rgbImage, rawDepthImage, objectName: str) -> dict|None:			
		if not self.isFresh_rgbd or type(rgbImage) == None or type(rawDepthImage) == None:
			return None
		
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
		(_, mask) = cv.threshold(norm_map, self.CONTOUR_THRESHOLD, 255, cv.THRESH_BINARY)
		(detail['contours'], _) = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)

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
		ex = (leftmost[0]+rightmost[0])//2
		ey = (topmost[1]+bottommost[1])//2
		detail['extremes_center'] = (ex, ey)

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

		# automatic object depth estimation using appropriate method
		if not math.isnan(rawDepthImage[cx, cy]) and not math.isnan(rawDepthImage[ex, ey]):
			# use the lowest depth value, to prevent any possible collisions/missalignments
			detail['obj_mid_depth'] = self._rawDepthToCm( min(rawDepthImage[cx, cy], rawDepthImage[ex, ey]) )
		else:
			# fallback to seg mean
			mean_region = np.array(rawDepthImage[leftmost[0]:rightmost[0], topmost[1]:bottommost[1]], dtype=np.float32)
			if(mean_region.size > 0):
				print('\t -- using seg mean depth')
				mask_bool = mask.astype(dtype=bool)
				nanFreeDepth = np.nan_to_num(rawDepthImage, nan=0.5, posinf=1.0, neginf=0.0)
				# self.vlm_depth_compute_frame = (nanFreeDepth * 255).round().astype(np.uint8) * mask
				mask_bool = ~mask_bool   # Invert the mask because True values are masked (ignored)
				masked_img = np.ma.array(nanFreeDepth, mask=mask_bool, dtype=np.float32)
				if math.isnan(rawDepthImage[ex, ey]):
					detail['obj_mid_depth'] = self._rawDepthToCm(masked_img.mean())
				else:
					detail['obj_mid_depth'] = self._rawDepthToCm(min(masked_img.mean(), rawDepthImage[ex, ey])) # take min of exey or mean
			else:
				detail['obj_mid_depth'] = math.nan
		
		print('> Object: %s, Mid: %.1fcm' % (objectName, detail['obj_mid_depth']))
   
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



	def _moveToHome(self) -> None:
		# moves the already grasped obj to home
		if not self.isEnacting:
			print("not enacting, skipping move to home")
			return None

		self.tkLabelCurrentText = 'Moving to home...'
		print("\tmoving to home")
		self.cartPose = (0.3, 0.3, 0.3, self.cartPose[3])
		self.cmndr.sendCommand(self.cartPose) # move to home pose, retain gripper value
		self.cmndr.awaitCommandFeedback()
		return None



	def _homeObjAndPlace(self, objPose: tuple, placePose: tuple, holdTimeSeconds = 2.0) -> None:
		# moves the already grasped obj to home, takes it back to pick location after N time and releases
		self._moveToHome()
		if not self.isEnacting:
			print("not enacting flagged, skipping homeObjAndPlace")
			return None
		
		self.tkLabelCurrentText = 'Holding for %.1fs' % holdTimeSeconds
		print('\t'+self.tkLabelCurrentText)
		time.sleep(holdTimeSeconds)
		self.isPlacing = True
		self.tkLabelCurrentText = 'Placing object...'
		print("\ttaking obj to place location")
		self.cartPose = placePose
		self.cmndr.sendCommand(self.cartPose) # move to place pose
		self.cmndr.awaitCommandFeedback()
		print("\treleasing obj at place location")
		self.cartPose = (placePose[0], placePose[1], placePose[2], 0.0)
		self.cmndr.sendCommand(self.cartPose) # release obj
		self.cmndr.awaitCommandFeedback()
		self.isPlacing = False
		return None



	def _getNextPoseStepXYZ(self, visionDetail: dict) -> tuple:
		(cx, cy) = visionDetail['bbox_center']
		(mx, my, tolPx) = visionDetail['gripper_center']
		mid_depth = visionDetail['obj_mid_depth']

		stepSize = 0.02 if mid_depth > 12 else 0.01

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

		if (xAligned and yAligned and mid_depth >= 1.5): # vertically aligned
			print('> --- vert aligned, decr z')
			stepz = -stepSize

		return (stepx, stepy, stepz, xAligned, yAligned)



	def pick(self, object_name: str, pick_and_return = True) -> None:
		if self.grasped:
			print("not picking, already grasped")
			return None
		
		print("\tinitiating pick obj: %s" % object_name)
		self.tkLabelCurrentText = 'Picking...'
		self.isEnacting = True
		self.isDetecting = True
		pickPose = None

		while (not self.endScript and not self.grasped and self.isEnacting):
			rgbImage = self.rgbd_frame[0]
			rawDepthImage = self.rgbd_frame[1]

			vision = self._applyAndGetVisionDetail(rgbImage, rawDepthImage, object_name)
			if vision == None:
				continue
			
			# get next pose step and alignment
			(stepx, stepy, stepz, xAligned, yAligned) = self._getNextPoseStepXYZ(vision)

			mid_depth = vision['obj_mid_depth']
			est_gripper = vision['grasp_estimate']

			if (xAligned and yAligned and mid_depth < 1.5): # vertically aligned and reached dest
				print('> --- grasp: %.1f%%' % (est_gripper*100,))
				self.gripper = est_gripper
				self.grasped = True

			nextPose = np.add(self.cartPose, (stepx, stepy, stepz, 0.0))
			nextPose[3] = self.gripper # gripper is set, not added
			self.cartPose = tuple(nextPose)
			if self.grasped:
				pickPose = tuple(nextPose)

			self.cmndr.sendCommand(self.cartPose)
			self.cmndr.awaitCommandFeedback() # awaits until command complete feedback is received

		if pick_and_return == False:
			return None

		# takes obj to home, holds for some time and brings back to pick location
		self._homeObjAndPlace(pickPose, pickPose, 10.0)
		self.resetFlagsAndParams()
		print('\taction complete')
		return True
	


	def pick_place(self, object_name: str, target_name: str) -> None:
		# STRATEGY: goto target location and save the pose, goto home, find and pick object, goto home, goto saved target pose

		if self.grasped:
			print("not doing pick_place, already grasped.")
			return None
		
		print("\tinitiating pick obj: %s and place at: %s" % (object_name, target_name))
		self.isEnacting = True
		self.isDetecting = True
		self.grasped = False
		self.gripper = 0.0
		targetPose = None

		self.tkLabelCurrentText = 'Finding place target...'
		while (not self.endScript and self.isEnacting and (targetPose) == None):
			rgbImage = self.rgbd_frame[0]
			rawDepthImage = self.rgbd_frame[1]
			vision = self._applyAndGetVisionDetail(rgbImage, rawDepthImage, target_name)
			if vision == None:
				continue
			# get next pose step and alignment
			(stepx, stepy, stepz, xAligned, yAligned) = self._getNextPoseStepXYZ(vision)
			self.cartPose = tuple(np.add(self.cartPose, (stepx, stepy, stepz, 0.0)))

			mid_depth = vision['obj_mid_depth']
			if (xAligned and yAligned and mid_depth < 1.5): # vertically aligned and reached dest
				targetPose = list(self.cartPose) # list because have to set gripper pose after picking
				print('> --- noted target pose (%.2f %.2f %.2f %.2f)' % self.cartPose)
				#loop terminates after this iteration

			self.cmndr.sendCommand(self.cartPose)
			self.cmndr.awaitCommandFeedback() # awaits until command complete feedback is received

		self.tkLabelCurrentText = 'Found target pose...'
		self._moveToHome() # take open gripper with nothing from target pose to home

		self.pick(object_name, pick_and_return=False)
		targetPose[3] = self.cartPose[3] # cartPose has pickPose, extract only gripper value
		self._homeObjAndPlace(self.cartPose, tuple(targetPose))
		self.resetFlagsAndParams()
		print('\taction complete')



	def ignore(self, *args) -> None:
		print("Do nothing")
