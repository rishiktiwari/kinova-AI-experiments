import socket
import threading
import json
import time
from datetime import datetime

class Commander:
	def __init__(self, HOST_IP: str):
		self.HOST_IP = HOST_IP
		self.HOST_PORT = 9999
		self.ENCODING_FORMAT = 'utf-8'
		self.HEADER_SIZE = 64
		self.DELIMITER = '[CMD]'
		self.LOG_DIR_PATH = "./data_recordings/"

		self.cmdlink_socket = None
		self.log_file = None
		self.log_file_name = ''
		self.script_start_time = None



	def initCommandLink(self, doManualCmd = False):
		print('Command link started')
		dt = datetime.now().strftime("%d-%b-%Y__%H-%M-%S")
		self.log_file_name = self.LOG_DIR_PATH + "actionLog__" + dt + ".csv"
		self.log_file = open(self.log_file_name, "a")
		print("--- writing to file: %s\n" % self.log_file_name)
		self.log_file.write('elapsedSec,x,y,z,gripper\n')

		self.script_start_time = time.perf_counter()
		self.cmdlink_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		socket_address = (self.HOST_IP, self.HOST_PORT)

		try:
			self.cmdlink_socket.connect(socket_address)
			print('Command connected to ', socket_address)
		except Exception as e: print(e)

		if doManualCmd:
			self.useManualCmd()
		else:
			print('using internal cmd mode')
			
		return None
	

	
	def _awaitCommandFeedback(self) -> str:
		# enables blocking
		try:
			print('[awaiting cmd feedback]')
			respMsg = self.cmdlink_socket.recv(self.HEADER_SIZE).decode(encoding=self.ENCODING_FORMAT)
			print('[cmd feedback: %s]' % respMsg)
			return respMsg
		
		except Exception as e:
			print('feedback err:', e)
		
		return ''



	def useManualCmd(self) -> None:
		while True:
			cmd = input('CMD: ')
			self.sendCommand(cmd)
			if(cmd == 'quit'):
				break
		self.closeLink()



	def sendCommand(self, command: str|tuple, awaitFeedback = False) -> str|None:
		if type(command) == tuple:
			# convert to string
			command = '%.2f %.2f %.2f %.2f' % command[:] # does not change original
		
		if (self.log_file != None and command != 'quit'):
			# CSV struct: elapsedSec, x, y, z, gripper
			csvEntry = str(round(time.perf_counter() - self.script_start_time, 3)) + ','
			csvEntry += command.replace(' ', ',') + '\n' # replace space with comma for valid CSV
			self.log_file.write(csvEntry)
			self.log_file.flush()

		command = json.dumps(obj={
			"pose": command,
			"feedback": awaitFeedback
		})
		print('sending cmd: \n|>\t%s' % command)
		command = command.encode(encoding=self.ENCODING_FORMAT) # max len is 50ch

		lengthDescriptor = (self.DELIMITER + str(len(command))).encode(encoding=self.ENCODING_FORMAT)
		lengthDescriptor += b'.' * (8 - len(lengthDescriptor)) # fixed 8 byte descriptor

		command = lengthDescriptor + command
		command += b'.' * (self.HEADER_SIZE - len(command)) # fixed HEADER_SIZE size, by adding trailing bytes

		try:
			if self.cmdlink_socket:
				self.cmdlink_socket.sendall(command)
				if awaitFeedback == True:
					return self._awaitCommandFeedback()
				return None
			
			else:
				raise Exception('socket unavailable!')
		except Exception as e:
			print(e)
			self.closeLink()


	
	def closeLink(self) -> None:
		print('Quitting commander...')
		if(self.log_file != None):
			self.log_file.close()
			print("\n\n--- closed file: %s\n\n" % self.log_file_name)

		if self.cmdlink_socket != None:
			try:
				self.cmdlink_socket.shutdown(socket.SHUT_RDWR)
				self.cmdlink_socket.close()
				self.cmdlink_socket = None
			except: pass
		print('Command link closed')
	


if (__name__ == '__main__'):
	print('COMMANDER: Starting in manual command mode.')
	cmndr = Commander()
	threading.Thread(target=cmndr.initCommandLink, kwargs={'doManualCmd': True}).start()
