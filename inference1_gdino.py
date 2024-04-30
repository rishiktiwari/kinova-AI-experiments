# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog

import socket
import pickle
import json
import math
import PIL
import PIL.Image
import torch
import threading
import re
import cv2 as cv
import numpy as np
# import tkinter as tk
import time

# llm stuff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_cpp import Llama

# dino stuff
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

# custom classes
from commander import Commander

class Inference1:
	def __init__(self):
		# self.HOST_IP = '192.168.1.114'
		# self.HOST_IP = '192.168.1.100'
		self.HOST_IP = '10.1.1.100'
		self.DATA_PORT = 9998
		self.CHUNK_SIZE = 1024 # increasing makes it unreliable
		self.ENCODING_FORMAT = 'utf-8'
		self.PACK_FLAG = "[PACKET]"
		self.TK_DEF_LABEL = "What can I pickup?"

		# self.window = tk.Tk()
		# self.window.title("Command Kinova Arm")
		self.tkLabelCurrentText = self.TK_DEF_LABEL
		# self.tkLabelVar = tk.StringVar(value=self.TK_DEF_LABEL)
		# tkLabel = tk.Label(textvariable=self.tkLabelVar, height=5, font=('Courier', '20', 'bold'))
		# tkLabel.pack()
		# self.humanCmdInput = tk.Entry(width=50, font=('Courier', '16'))
		# self.humanCmdInput.pack()
		# guiSendLambda = lambda evt: self.getCmdEntryVal() # bcoz the btn does not send self when called directly
		# sendBtn = tk.Button(text="Send", command=guiSendLambda).pack()
		# self.window.bind('<Return>', guiSendLambda)

		self.cartPose = (0.3, 0.3, 0.3, 0.0)
		self.cmndr = Commander()
		self.cmndr.initCommandLink()
		self.cmndr.sendCommand('%.2f %.2f %.2f %.2f' % self.cartPose[:])

		self.device = 'cpu'
		self.endScript = False
		self.isDetecting = False
		self.grasped = False
		self.isPlacing = False
		self.gripper = 0.0
		self.llmIntel = None
		self.noObjFrames = 0 # num of frames with no detected objs
		
		self.checkAIDevice()
		# self.llm = Llama(
		# 	model_path='/Volumes/Rishik T7_1/AI/mistralChatbot/models/llama2/llama-2-7b-chat.Q4_K_M.gguf',
		# 	n_gpu_layers=-1,
		# 	temperature=0.4,
		# 	top_k=40,
		# 	top_p=0,
		# 	repeat_penalty=1.17,
		# 	n_batch=1024,
		# 	n_ctx=2048,
		# 	chat_format='llama-2'
		# )
		self.gDinoModel = load_model('groundingdino/config/GroundingDINO_SwinT_OGC.py', 'gDinoWeights/groundingdino_swint_ogc.pth')
		self.annotated_dino_frame = np.zeros(shape=(10,10), dtype=np.uint8)
		self.annotated_dino_depthframe = np.zeros(shape=(10,10), dtype=np.uint8)

		cv.namedWindow('rgb_preview', cv.WINDOW_AUTOSIZE)
		cv.moveWindow('rgb_preview', 100, 200)
		cv.namedWindow('depth_preview', cv.WINDOW_AUTOSIZE)
		cv.moveWindow('depth_preview', 420, 200)
		cv.namedWindow('annotation_preview', cv.WINDOW_AUTOSIZE)
		cv.moveWindow('annotation_preview', 840, 200)
		# cv.namedWindow('annotation_depth_preview', cv.WINDOW_AUTOSIZE)
		# cv.moveWindow('annotation_depth_preview', 1160, 200)

		return None
	


	def checkAIDevice(self):
		# Check that MPS is available
		if not torch.backends.mps.is_available():
			if not torch.backends.mps.is_built():
				print("MPS not available because the current PyTorch install was not built with MPS enabled.")
			else:
				print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")

		else:
			print('MPS is available and selected!')
			self.device = torch.device("mps")

		torch.set_default_device(self.device)
		return None



	# def getCmdEntryVal(self):
	# 	if(self.llmIntel != None):
	# 		print('A command is already active, please wait for it to finish!')
	# 	else:
	# 		cmd = self.humanCmdInput.get().strip()
	# 		if(cmd != ''):
	# 			print('--> HUMAN COMMAND REQUEST: %s' % cmd)
	# 			# self.tkLabelVar.set('Thinking...')
	# 			# self.window.update()
	# 			self.llmIntel = self.getLlmIntel(cmd)
	# 			if(self.llmIntel == None):
	# 				print('Please try again :(')
	# 				# self.tkLabelVar.set('Please say that again')
	# 				# self.window.update()
		
	# 	self.humanCmdInput.delete(0, tk.END)



	def inferTask(self, taskMsg): # LLM Inference
		systemInstruct = '''
		Your job is to extract action and objects from the given task. Respond only in JSON format.
		\nDo not hello, comment, explain, or elaborate your thought processes. Your response should have two keys: actions and objects. Split the task in as many as sub-actions possible. Only mention the name of object. curly brackets are important.
		\n
		\nExample 1:
		\nTask: Pick up banana and place on the plate.
		\nAnswer: { "actions": ["pick", "place"], objects: ["banana", "plate"] }
		'''
		userMsg = 'Task: ' + str.strip(taskMsg)

		llmOutput = self.llm.create_chat_completion(
			messages = [
				{"role": "system", "content": systemInstruct},
				{
					"role": "user",
					"content": userMsg
				}
			],
			# stop=["<|endoftext|>"]
		)

		cleanOutput = str.strip(llmOutput['choices'][0]['message']['content'])
		cleanOutput = re.sub('[^A-Za-z0-9\"\':\{\}\[\]\_\-\,]+', '', cleanOutput)

		return cleanOutput



	def getLlmIntel(self, taskMsg: str) -> dict|None:
		isSatisfiableLlmOutput = False
		llmAttemptCounter = 0
		llmIntel = None

		while ((isSatisfiableLlmOutput == False) and (llmAttemptCounter <= 5)):
			llmAttemptCounter += 1
			llmIntel = self.inferTask(taskMsg)
			print('LLM Intel #%d: %s' % (llmAttemptCounter, llmIntel))
			if llmIntel:
				try:
					llmIntel = json.loads(llmIntel)
					isSatisfiableLlmOutput = True
					break
				except:
					break

		if not isSatisfiableLlmOutput:
			print('Could not get a satisfiable answer from llm.')
			llmIntel = None

		print(f'LLM attempts: %d' % llmAttemptCounter)
		print(llmIntel)
		return llmIntel



	def loadImgForGDino(self, img: PIL.Image.Image):
		transform = T.Compose(
			[
				T.RandomResize([800], max_size=1333),
				T.ToTensor(),
				T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			]
		)
		image_source = img.convert("RGB")
		image = np.asarray(image_source)
		image_transformed, _ = transform(image_source, None)
		return image, image_transformed
	


	def detectObjectsInImage(self, rgbImage, rawDepthImage):
		if self.grasped:
			return None
		
		self.tkLabelCurrentText = 'Picking...'
		self.isDetecting = True
		TEXT_PROMPT = ' . '.join(self.llmIntel['objects']) # all llmIntel objects are joined by dot
		BOX_TRESHOLD = 0.4
		TEXT_TRESHOLD = 0.3
		GRIPPER_LENGTH_CM = 22

		rgbFrameImg = PIL.Image.fromarray(rgbImage[:,:,::-1], mode='RGB')
		rgbImg_source, rgbImg = self.loadImgForGDino(rgbFrameImg)
		# print('DINO prompt: %s' % TEXT_PROMPT)

		start_time = time.perf_counter()
		dino_boxes, dino_logits, dino_phrases = predict(
			model=self.gDinoModel,
			image=rgbImg,
			caption=TEXT_PROMPT,
			box_threshold=BOX_TRESHOLD,
			text_threshold=TEXT_TRESHOLD,
			device="cpu"                #NOTE: cpu and mps have different predictions
		)
		end_time = time.perf_counter()
		print('DINO inference time %.2fms' % ((end_time-start_time)*1000))
		
		self.annotated_dino_frame = annotate(image_source=rgbImg_source, boxes=dino_boxes, logits=dino_logits, phrases=dino_phrases)
		
		(h, w) = rgbImg_source.shape[:2]
		mx = (h//4)*3   # shift mid point to lower half near the gripper
		my = w//2
		# the boxes value is in format: cxcywh - box center x, center y, width, height
		normalised_bboxes = np.round(np.array(dino_boxes * torch.Tensor([w, h, w, h]), dtype=np.uint8), decimals=0)
		
		numOfItem = len(normalised_bboxes)
		# numOfConfidentItems = 0
		colour = (0, 255, 0)
		tolPx = 20
		for i in range(numOfItem):
			# if(dino_logits[i] <= 0.7):
			# 	continue
			# numOfConfidentItems += 1
   
			(cx, cy, bw, bh) = normalised_bboxes[i][:]
			mid_depth = rawDepthImage[cy, cx] * 100 - (GRIPPER_LENGTH_CM - 2.5) # incl offset correction
			cv.circle(self.annotated_dino_frame, (cx, cy), radius=5, color=(255,0,255), thickness=-1) # obj mid point
			
			# fallback to mean if mid is nan
			if (math.isnan(mid_depth)):
				print('>> using mean depth <<')
				mean_region = rawDepthImage[cy-bh//2:cy+bh//2, cx-bw//2:cx+bw//2]
				mid_depth = np.nanmean(mean_region) * 100 - (GRIPPER_LENGTH_CM - 2.5)

			print('> Object: %s, Mid: %.1fcm' % (dino_phrases[i], mid_depth))

			stepx = 0.0
			stepy = 0.0
			stepz = 0.0

			xAligned = False
			yAligned = False

			if(cx > my+tolPx):
				stepx = 0.02
			elif(cx < my-tolPx):
				stepx = -0.02
			else:
				xAligned = True
			
			if(cy > mx+tolPx):
				stepy = -0.02
			elif(cy < mx-tolPx):
				stepy = 0.02
			else:
				yAligned = True

			if (xAligned and yAligned):
				if (mid_depth >= 2):
					print('> --- decr z')
					stepz = -0.02
				elif (mid_depth < 2):
					print('> --- grasp')
					self.gripper = 0.5 # TODO: estimate obj width from segmentation mask
					self.grasped = True

			step = np.add(self.cartPose, (stepx, stepy, stepz, 0.0))
			step[3] = self.gripper # gripper is fixed, not added
			self.cartPose = tuple(step)

		if(numOfItem == 0):
			self.noObjFrames += 1
			if self.noObjFrames == 3:
				print('--- no objs, sending to home ---')
				self.cartPose = (0.3, 0.3, 0.3, 0.0)
				self.noObjFrames = 0

		self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % self.cartPose[:])

		cv.line(self.annotated_dino_frame, (0, mx), (w, mx), colour, 1) # horz
		cv.line(self.annotated_dino_frame, (my, 0), (my, h), colour, 1) # vert
		cv.rectangle(self.annotated_dino_frame, (my-tolPx, mx-tolPx), (my+tolPx, mx+tolPx), color=(0,255,255), thickness=1) # tolerance region
		
		depthNormalised = self.normaliseDepthForPreview(rawDepthImage, True)
		self.annotated_dino_depthframe = annotate(image_source=depthNormalised, boxes=dino_boxes, logits=dino_logits, phrases=dino_phrases)
		self.isDetecting = False
		return True
	


	def executeDemoPlace(self):
		if self.grasped:
			print('--- executing sample place ---')
			self.tkLabelCurrentText = 'Placing object...'
			self.isPlacing = True
			# already has grasped, no object detection required, executing primitive for example
			self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % (self.cartPose[0], self.cartPose[1], 0.3, self.gripper)) # raise vert at pick pose
			time.sleep(2)
			self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % (0.4, 0.1, 0.3, self.gripper)) # align vert to place pose
			time.sleep(2)
			self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % (0.4, 0.1, self.cartPose[2], self.gripper)) # dest pose, z from pickup history
			time.sleep(2)
			self.gripper = 0.0
			self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % (0.4, 0.1, self.cartPose[2], self.gripper)) # place
			time.sleep(2)
			self.cartPose = (0.3, 0.3, 0.3, self.gripper)
			self.cmndr.sendCommand('%.3f %.3f %.3f %.2f' % self.cartPose[:]) # home the robot
			time.sleep(20) # let the action complete
			self.grasped = False
			self.isPlacing = False
			self.llmIntel = None
			self.tkLabelCurrentText = self.TK_DEF_LABEL
			print('--- obj demo place complete ---')
			return None



	def normaliseDepthForPreview(self, rawDepthImage, reshapeOneChannel = False):
		depthFrameNormalised = np.nan_to_num(rawDepthImage, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
		depthFrameNormalised = (depthFrameNormalised * 255).round().astype(np.uint8)

		if reshapeOneChannel:
			# reshape from WxH to WxHxC
			depthFrameNormalised = np.reshape(depthFrameNormalised, (depthFrameNormalised.shape[0], depthFrameNormalised.shape[1], 1))
		
		return depthFrameNormalised


	def getWordIndex(self, word: str, text: str, skip: int = 0):
		i = -1
		try:
			i = str(text).index(word, skip)
		except ValueError:
			pass
		return i



	def initDataLink(self):
		print('Data link started')
		datalink_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		socket_address = (self.HOST_IP,self.DATA_PORT)

		self.llmIntel = {'objects': ['ball']}

		try:
			datalink_socket.connect(socket_address)
			print('Connected to host: ', socket_address)

			data = ''
			pack_start = -1
			pack_end = -1
			while not self.endScript:
				# self.tkLabelVar.set(self.tkLabelCurrentText) # refreshes label text
				# self.window.update()

				# await data length message
				data += datalink_socket.recv(self.CHUNK_SIZE).decode(encoding=self.ENCODING_FORMAT)

				pack_start = self.getWordIndex(self.PACK_FLAG, data)
				pack_end = self.getWordIndex(self.PACK_FLAG, data, pack_start+8) if (pack_start != -1) else -1

				if(pack_start != -1 and pack_end > pack_start):
					package = data[pack_start+self.CHUNK_SIZE : pack_end]; # skips first chunk - length descriptor
					# print(package)
					data = data[pack_end:] # remove this package from buffer
					print('Full data rcvd, size: %d, in-buf: %d' % (len(package), len(data)))
				else:
					continue

				try:
					# full_data = pickle.loads(full_data, encoding='latin1')
					# full_data = full_data.decode(encoding=self.ENCODING_FORMAT)
					package = json.loads(package)
					# print(full_data['pos'])
					
					(rgbImage, depthImage) = pickle.loads(bytes(package['rgbd'], self.ENCODING_FORMAT), encoding='latin1')
					# rgbImage = np.array(rgbdImage[:, :, 0:3], dtype=np.uint8)
					# depthImage = np.array(rgbdImage[:, :, 3])
					
					# midDist = (depthImage[160][120] - 0.225)*100
					# print('mid dist: %.1fcm' % midDist)

					cv.imshow('rgb_preview', rgbImage)
					cv.imshow('depth_preview', self.normaliseDepthForPreview(depthImage))
					# if self.annotated_dino_frame:
					# 	print('refreshing annotated frame')
					cv.imshow('annotation_preview', np.concatenate((self.annotated_dino_frame, self.annotated_dino_depthframe), axis=1))
					# cv.imshow('annotation_preview', self.annotated_dino_frame)
					# cv.imshow('annotation_depth_preview', self.annotated_dino_depthframe)
					cv.waitKey(1)

				except Exception as e:
					print(e)
					print('data parse/preview failed')

				if (self.llmIntel != None and not self.isDetecting and not self.isPlacing):
					if self.grasped:
						threading.Thread(target=self.executeDemoPlace).start()
					else:
						threading.Thread(target=self.detectObjectsInImage, args=(rgbImage, depthImage)).start()

		except Exception as e:
			print(e)
			print('Data link failed, closing')
		except KeyboardInterrupt:
			print('Keyboard interrupt')
		finally:
			self.endScript = True
			# datalink_socket.sendall('quit')
			self.cmndr.sendCommand('quit') # closes remote socket
			self.cmndr.closeLink()
			datalink_socket.shutdown(socket.SHUT_RDWR)
			datalink_socket.close()



if __name__ == '__main__':
	print("Connecting to all links..")
	inf = Inference1()
	inf.initDataLink()
