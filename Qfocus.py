# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 03:14:52 2018

@author: shaksuma
"""

#python Qfocus.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import datetime
import cv2
import pyaudio  
import wave
import matplotlib.pylab as plt

#define stream chunk   
chunk = 1024
f_hours = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0,22:0,23:0}
ct = 0
counter = 0
msg = ""

def showGraph():
	print f_hours
	lists = sorted(f_hours.items())
	x, y = zip(*lists)
	plt.plot(x, y)
	plt.show()
	
def play_sound():
	#open a wav format music 
	if counter > 50:
		f_hours[int(time.strftime("%H"))] = f_hours.get(int(time.strftime("%H"))) + 1
		print msg 
		f = wave.open("DING.wav","rb")  
		#instantiate PyAudio  
		p = pyaudio.PyAudio()  
		#open stream  
		stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
		                channels = f.getnchannels(),  
		                rate = f.getframerate(),  
		                output = True)  
		#read data  
		data = f.readframes(chunk)  

		#play stream  
		while data:  
		    stream.write(data)  
		    data = f.readframes(chunk)  

		#stop stream  
		stream.stop_stream()  
		stream.close()  

		#close PyAudio  
		p.terminate()   

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

vs = VideoStream(src=0).start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < args["confidence"]:
			continue
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		if (confidence*100) < 98.0:
			ts=time.time()
			print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d  %H:%M:%S')
			counter = counter + 1
			msg = "  Not Focused"
			play_sound()
		else:
			counter = 0
			ts=time.time()
			print datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d  %H:%M:%S')
			msg = "  Focused"
		
		#print msg

		text = "{:.2f}%".format(confidence * 100) + msg
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 50, 0), 2)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		showGraph()
		break

cv2.destroyAllWindows()
vs.stop()