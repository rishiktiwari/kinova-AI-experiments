import socket, pickle, json
import cv2 as cv
import numpy as np

# HOST_IP = '192.168.1.114'
HOST_IP = '10.1.1.100'
DATA_PORT = 9998
CHUNK_SIZE = 1024 # increasing makes it unreliable
ENCODING_FORMAT = 'utf-8'
PACK_FLAG = "[PACKET]"

endScript = False

def getWordIndex(word, text, skip = 0):
	i = -1
	try:
		i = str(text).index(word, skip)
	except ValueError:
		pass
	return i

def data_link():
	global endScript
	print('Data link started')
	datalink_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	socket_address = (HOST_IP, DATA_PORT)

	# subsampled_skip = 20
	# subsample_counter = 0

	try:
		datalink_socket.connect(socket_address)
		print('Connected to host: ', socket_address)

		data = ''
		while True and (not endScript):
			data += datalink_socket.recv(CHUNK_SIZE).decode(encoding=ENCODING_FORMAT)

			pack_start = getWordIndex(PACK_FLAG, data)
			pack_end = getWordIndex(PACK_FLAG, data, pack_start+8) if (pack_start != -1) else -1

			# print("s: %d, e: %d, buf_size: %d" % (pack_start, pack_end, len(data)))

			if(pack_start != -1 and pack_end > pack_start):
				package = data[pack_start+CHUNK_SIZE : pack_end]; # skips first chunk - length descriptor
				rcvdPackSize = len(package)
				print('Full pack rcvd, size: %d' % rcvdPackSize)
				# print(package)

				data = data[pack_end:] # remove this package from buffer

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

	except Exception as e:
		print(e)
		print('Data link failed, closing')
	except KeyboardInterrupt:
		pass

	endScript = True
	datalink_socket.shutdown(socket.SHUT_RDWR)
	datalink_socket.close()


print("Starting data link...")
cv.namedWindow('rgb_preview', cv.WINDOW_AUTOSIZE)
cv.namedWindow('depth_preview', cv.WINDOW_AUTOSIZE)
data_link()
