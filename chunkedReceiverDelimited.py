import socket, pickle, json
import cv2 as cv
# import numpy as np
from time import sleep
import threading

HOST_IP = '192.168.1.100'
# HOST_IP = '192.168.1.114'
# HOST_IP = '10.1.1.100'
DATA_PORT = 9998
CHUNK_SIZE = 1024 # increasing makes it unreliable
ENCODING_FORMAT = 'utf-8'
PACK_FLAG = "[PACKET]"

endScript = False

# def getWordIndex(word: str, text: str, skip = 0):
# 	i = -1
# 	try:
# 		i = str(text).index(word, skip)
# 	except ValueError:
# 		pass
# 	return i


def handleUserCmd(datalink_socket):
	while not endScript:
		cmd = input('Command [frame/quit]: ')
		datalink_socket.sendall(cmd.encode(encoding=ENCODING_FORMAT))
		sleep(1.0)

	return None



def data_link():
	global endScript
	print('Data link started')
	datalink_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	socket_address = (HOST_IP, DATA_PORT)

	# subsampled_skip = 20
	# subsample_counter = 0

	try:
		print('Awaiting connection...')
		datalink_socket.connect(socket_address)
		print('Connected to host: ', socket_address)

		threading.Thread(target=handleUserCmd, args=(datalink_socket,)).start()

		pack_delimiter_len = len(PACK_FLAG)
		buffer = ''
		buffer_len = 0
		while True and (not endScript):
			buffer += datalink_socket.recv(CHUNK_SIZE).decode(encoding=ENCODING_FORMAT)
			buffer_len = len(buffer)
			pack_start = buffer.find(PACK_FLAG, 0)
			# print("s: %d, e: %d, buf_size: %d" % (pack_start, pack_end, len(buffer)))

			if (pack_start != -1 and buffer_len > pack_start+CHUNK_SIZE): # one complete chunk exists containing buffer len describer exists
				total_data_len = int(buffer[pack_start+pack_delimiter_len : CHUNK_SIZE].strip('.')) # extract total buffer len from first chunk of fixed CHUNK_SIZE in following format [delimiter]123....  
				pack_end = pack_start+CHUNK_SIZE+total_data_len

				if (buffer_len >= pack_end):
					package = buffer[pack_start+CHUNK_SIZE : pack_end]; # skips first chunk - length descriptor
					rcvdPackSize = len(package)

					print('---full data rcvd %d' % rcvdPackSize)
					buffer = buffer[pack_end:] # remove this package from buffer
					print('in buf: %d' % len(buffer))

					try:
						# full_data = pickle.loads(full_data, encoding='latin1')
						package = json.loads(package)
						# print(full_data['pos'])
						
						(rgbImage, depthImage) = pickle.loads(bytes(package['rgbd'], ENCODING_FORMAT), encoding='latin1')

						cv.imshow('rgb_preview', rgbImage)
						cv.imshow('depth_preview', depthImage)
						cv.waitKey(1)

					except Exception as e:
						print(e)
						print('data parse/preview failed')

	except Exception or KeyboardInterrupt as e:
		print(e)
		print('Data link failed, closing')
	finally:
		endScript = True
		datalink_socket.sendall('quit'.encode(encoding='utf-8'))
		sleep(0.5)
		datalink_socket.shutdown(socket.SHUT_RDWR)
		datalink_socket.close()


print("Starting data link...")
cv.namedWindow('rgb_preview', cv.WINDOW_AUTOSIZE)
cv.namedWindow('depth_preview', cv.WINDOW_AUTOSIZE)
data_link()
