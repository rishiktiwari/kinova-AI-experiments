import socket
import threading

class Commander:
	def __init__(self):
		# self.HOST_IP = '192.168.1.114'
		# self.HOST_IP = '192.168.1.100'
		self.HOST_IP = '10.1.1.100'
		self.HOST_PORT = 9999
		self.ENCODING_FORMAT = 'utf-8'

		self.cmdlink_socket = None


	def initCommandLink(self, doManualCmd = False):
		print('Command link started')
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
	


	def awaitCommandFeedback(self) -> str:
		# not continuous to enable blocking, call when required
		try:
			print('[awaiting cmd feedback]')
			respMsg = self.cmdlink_socket.recv(8).decode(encoding=self.ENCODING_FORMAT)
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



	def sendCommand(self, command: str|tuple):
		if type(command) == tuple:
			# convert to string
			command = '%.2f %.2f %.2f %.2f' % command[:] # does not change original

		try:
			if self.cmdlink_socket:
				print('sending cmd: \t'+command)
				self.cmdlink_socket.sendall(command.encode(encoding=self.ENCODING_FORMAT))
			else:
				raise Exception('socket unavailable!')
		except Exception as e:
			print(e)
			self.closeLink()


	
	def closeLink(self) -> None:
		print('Quitting commander...')
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
	# cmndr.initCommandFeedbackReceiver()
