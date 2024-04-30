import socket
import pickle
import json
import torch
import threading
import re
import cv2 as cv
import numpy as np
import tkinter as tk
import time
import ast
from queue import Queue

# llm stuff
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_cpp import Llama

# custom class
from primitiveActions import PrimitiveActions



class Inference1(PrimitiveActions):
	def __init__(self):
		super().__init__()
		# self.HOST_IP = '192.168.1.114'
		# self.HOST_IP = '192.168.1.100'
		self.HOST_IP = '10.1.1.100'
		self.DATA_PORT = 9998
		self.CHUNK_SIZE = 1024 # increasing makes it unreliable
		self.ENCODING_FORMAT = 'utf-8'
		self.PACK_FLAG = "[PACKET]"

		self.window = tk.Tk()
		self.window.title("Command Kinova Arm")
		self.window.geometry('+690+100')
		self.tkLabelVar = tk.StringVar(value=self.TK_DEF_LABEL)
		tkLabel = tk.Label(textvariable=self.tkLabelVar, height=5, font=('Courier', '20', 'bold'))
		tkLabel.pack()
		self.humanCmdInput = tk.Entry(width=50, font=('Courier', '16'))
		self.humanCmdInput.pack()
		tk.Button(text="Send", command=self.getCmdEntryVal).pack()
		tk.Button(text="Cancel Task", command=self.resetFlagsAndParams).pack()
		self.window.bind('<Return>', self.getCmdEntryVal)

		self.socket_buffer = ''
		self.package_queue = Queue()
		
		# self.checkAIDevice()
		self.llm = Llama(
			model_path="./llm_models/Phi-3-mini-4k-instruct-q4.gguf",
			n_ctx=4096,  # The max sequence length to use
			n_threads=8,
			n_gpu_layers=-1
		)

		cv.namedWindow('arm_vision', cv.WINDOW_AUTOSIZE)
		cv.moveWindow('arm_vision', 100, 100)
		cv.namedWindow('vlm_preview', cv.WINDOW_AUTOSIZE)
		cv.moveWindow('vlm_preview', 100, 420)

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



	def isValidPython(self, code: str) -> bool:
		# adapted from https://stackoverflow.com/a/11854793/4036127 and https://stackoverflow.com/a/51271704/4036127
		try:
			m = ast.parse(code)
			m_args = [a.value for a in m.body[0].value.args]
			id_name = m.body[0].value.func.value.id
			def_name = m.body[0].value.func.attr

			# hardcoded and assumed class object is in variable named: action
			if (id_name != 'action' or len(m_args) != PrimitiveActions.REQUIRED_ARGS[def_name]):
				return False
		except Exception as e:
			print('pythonic check err: ', e)
			return False
		return True



	def getCmdEntryVal(self, evt = None):
		if(self.llmIntel != None):
			print('A command is already active, please wait for it to finish!')
		else:
			cmd = self.humanCmdInput.get().strip()
			if(cmd != ''):
				print('--> HUMAN COMMAND REQUEST: %s' % cmd)
				self.tkLabelCurrentText = 'Thinking...'
				self.window.update()
				threading.Thread(target=self.askLLM, args=(cmd,)).start()
		
		self.humanCmdInput.delete(0, tk.END)



	def inferTask(self, taskMsg) -> list|None: # LLM Inference
		cleanOutput = None
		prompt = '''
			Your job is to extract actions and objects from the given task and invoke the allowed functions sequentially. Answer only with action.ignore() if no appropriate function available or you don't know what to do.
			Do not greet, comment, explain, or elaborate your thought processes. Split the task in as many as sub-actions possible. Use semicolon to seperate the function calls.
			Always recognise task language and translate the task and the object name to English.
			Allowed functions are: action.pick(), action.pick_place(), action.ignore()
			\nExample 1:
			Task: Pick up banana and place on the plate.
			Answer: action.pick_place("banana", "plate")
			\nExample 2:
			Task: Pickup the blue box.
			Answer: action.pick("blue box")
			\nNow translate to English and answer for Task: ''' + taskMsg

		start_time = time.perf_counter()
		llmOutput = self.llm(
			f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
			max_tokens=160,  # Generate up to 256 tokens
			stop=["<|end|>"],
			echo=False,
			temperature=0.3
		)
		end_time = time.perf_counter()
		print("LLM inference time: %.1fms\n" % ((end_time-start_time)*1000,))

		try:
			cleanOutput = str(llmOutput['choices'][0]['text'])
			newLineIdx = cleanOutput.find('\n')
			if newLineIdx != -1:
				cleanOutput = cleanOutput[:newLineIdx]

			cleanOutput = re.sub('[^A-Za-z0-9\"\':\{\}\[\]\_\-\,\.\(\)\;\s]+', '', cleanOutput).strip()
			cleanOutput = cleanOutput.split(';')
			cleanOutput = [a.strip() for a in cleanOutput if a != ''] # filters empty strings and removes edge whitespaces

		except Exception as e:
			print('llm output parsing err:', e)
			cleanOutput = None
		
		return cleanOutput



	def askLLM(self, taskMsg: str) -> None:
		isSatisfiableLlmOutput = False
		llmAttemptCounter = 0
		actionCount = 0
		llmIntel = None

		while ((isSatisfiableLlmOutput == False) and (llmAttemptCounter < 5)):
			llmAttemptCounter += 1
			llmIntel = self.inferTask(taskMsg)
			if not llmIntel:
				print('LLM Intel #%d, got nothing' % llmAttemptCounter)
				continue

			print('LLM Intel attempt #%d: ' % llmAttemptCounter, '\n\t', llmIntel)
			actionCount = len(llmIntel)
			validActions = 0
			for act in llmIntel: # test each action: str is pythonic or not.
				validActions += 1 if self.isValidPython(act) else 0
			
			print('Valid LLM intels: %d/%d' % (validActions, actionCount))
			if validActions == actionCount: # all actions are valid and pythonic
				isSatisfiableLlmOutput = True
				break

		if isSatisfiableLlmOutput:
			# create queue and all actions to it sequentially
			self.llmIntel = Queue(maxsize=actionCount)
			for act in llmIntel:
				self.llmIntel.put(act)
		else:
			print('Could not get a satisfiable answer from llm.')
			self.tkLabelCurrentText = self.TK_DEF_LABEL
		return None



	def getWordIndex(self, word: str, text: str, skip: int = 0):
		i = -1
		try:
			i = str(text).index(word, skip)
		except ValueError:
			pass
		return i



	def cleanExit(self):
		self.endScript = True
		self.resetFlagsAndParams()
		self.cmndr.sendCommand('quit') # closes remote socket
		self.cmndr.closeLink()



	def initDataLink(self):
		time.sleep(1)
		print('Data link started')
		datalink_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		socket_address = (self.HOST_IP,self.DATA_PORT)

		try:
			datalink_socket.connect(socket_address)
			print('Connected to host: ', socket_address)

			pack_start = -1
			pack_end = -1
			while not self.endScript:
				# await data length message
				self.socket_buffer += datalink_socket.recv(self.CHUNK_SIZE).decode(encoding=self.ENCODING_FORMAT)

				pack_start = self.getWordIndex(self.PACK_FLAG, self.socket_buffer)
				pack_end = self.getWordIndex(self.PACK_FLAG, self.socket_buffer, pack_start+8) if (pack_start != -1) else -1

				if(pack_start != -1 and pack_end > pack_start):
					package = self.socket_buffer[pack_start+self.CHUNK_SIZE : pack_end]; # skips first chunk - length descriptor
					# print(package)
					self.socket_buffer = self.socket_buffer[pack_end:] # remove this package from buffer
					print('Full data rcvd, size: %d, in-buf: %d' % (len(package), len(self.socket_buffer)))
					self.package_queue.put(package)
				else:
					continue

		except Exception as e:
			print(e)
			print('Data link failed, closing')
		except KeyboardInterrupt:
			print('Keyboard interrupt')
		finally:
			print('Closing...')
			# datalink_socket.sendall('quit')
			datalink_socket.shutdown(socket.SHUT_RDWR)
			datalink_socket.close()
			self.cleanExit()



	def initServices(self):
		action = PrimitiveActions()

		while not self.endScript:
			self.tkLabelVar.set(self.tkLabelCurrentText) # refreshes label text
			self.window.update()
			
			package = None
			if not self.package_queue.empty():
				package = self.package_queue.get()
			else:
				continue

			try:
				package = json.loads(package)
				rgbd = pickle.loads(bytes(package['rgbd'], self.ENCODING_FORMAT), encoding='latin1')
				# midDist = (depthImage[160][120] - 0.225)*100
				# print('mid dist: %.1fcm' % midDist)
				
				# self.rgbd_frame_queue.put(rgbd, block=True, timeout=None)
				self.rgbd_frame = rgbd
				self.isFresh_rgbd = True

				cv.imshow(
					'arm_vision',
					np.concatenate((rgbd[0], self.normaliseDepthForPreview(rgbd[1], True)), axis=1)
				)
				cv.imshow('vlm_preview', np.concatenate(
					(
						self.vlm_ann_rgbframe,
						np.repeat(self.vlm_heatmap[:, :, np.newaxis], 3, axis=2)
					),
					axis=1
				))
				cv.waitKey(1)

			except Exception as e:
				print(e)
				print('data parse/preview failed')

			if (not self.isEnacting and not self.endScript and type(self.llmIntel) == Queue):
				print('action available')	 
				if self.llmIntel.qsize() != 0:
					print('starting action thread...')
					action_name = self.llmIntel.get()
					action_module = ast.parse(action_name)
					action_def_name = action_module.body[0].value.func.attr
					action_args = tuple([a.value for a in action_module.body[0].value.args]) # extracts and creates a tuple of args to pass to action method
					action_def = getattr(self, action_def_name) # creates a reference to action method without invoking and stores in action_def

					# start the primitive action function as thread
					threading.Thread(target=action_def, args=action_args, daemon=False).start()
			
			time.sleep(0.1)



	def launch(self):
		threading.Thread(target=self.initDataLink).start()
		self.initServices()



if __name__ == '__main__':
	print("Connecting to all links..")
	inf = Inference1()
	inf.launch()
