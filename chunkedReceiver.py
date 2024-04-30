import socket, pickle, json
import cv2 as cv
import numpy as np

# HOST_IP = '192.168.1.114'
HOST_IP = '10.1.1.100'
DATA_PORT = 9998
CHUNK_SIZE = 1024 # increasing makes it unreliable
ENCODING_FORMAT = 'utf-8'

endScript = False



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

		while True and (not endScript):
			# await data length message
			data = b''
			while(len(data) < CHUNK_SIZE):
				data += datalink_socket.recv(CHUNK_SIZE)
			
			data = data.decode(encoding=ENCODING_FORMAT)

			# extract upcoming data length
			try:
				# print('--rcvd length descriptor--')
				total_data_length = int(data.strip('.'))
				# print('pkg len %d' %total_data_length)
			except Exception as e:
				print('length extraction failed')
				continue

			print('Data Len: %d' % total_data_length)

			# receive the chunked data and concatenate
			print('Receiving...')
			full_data = b''
			rcvd_len = 0

			while (rcvd_len < total_data_length):
				chunk = datalink_socket.recv(CHUNK_SIZE)
				# print(chunk)
				if (not chunk):
					raise Exception('empty chunk rcvd')
				
				full_data += chunk
				rcvd_len = len(full_data)
				# print('Rcvd %d / %d' % (rcvd_len, total_data_length))
			
			print('-- Full Data Rcvd, size: %d --' % rcvd_len)

			try:
				# full_data = pickle.loads(full_data, encoding='latin1')
				full_data = full_data.decode(encoding=ENCODING_FORMAT)
				full_data = json.loads(full_data)
				# print(full_data['pos'])
				
				(rgbImage, depthImage) = pickle.loads(bytes(full_data['rgbd'], ENCODING_FORMAT), encoding='latin1')
				# rgbImage = np.array(rgbdImage[:, :, 0:3], dtype=np.uint8)
				# depthImage = np.array(rgbdImage[:, :, 3])
				
				# midDist = (depthImage[160][120] - 0.225)*100
				# print('mid dist: %.1fcm' % midDist)

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
