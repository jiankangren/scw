#!/usr/bin/python
import sys, subprocess, socket, serial, logging#,obd
import time, threading
from threading import Timer,Thread,Event
from time import sleep


class perpetualTimer():

   def __init__(self,t,hFunction):
      self.t=t
      self.hFunction = hFunction
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('tx.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s , %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

subprocess.Popen( "./setNetwork.sh 1", shell=True)
sleep(1)

UDP_IP = "192.168.1.2"
UDP_PORT = 5005
MESSAGE = "NODATA"

ser = serial.Serial('/dev/ttyUSB0', 4800, timeout=1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#conn = obd.OBD()
#c = obd.commands[1][13]
speed = 'NODATA'
seqNo = 0

def readOBD():
    global speed
    #speed = conn.query(c)

def readGPS():
    global latitude, longitude
    latitude = 'NODATA'
    longitude = 'NODATA'
    line = ser.readline()
    if "GPGGA" in line:
        latitude = line[18:26]
        longitude = line[31:39]
	    

def timerHandler():
    global seqNo, longitude, latitude, speed
    MESSAGE = str(seqNo) +',' + str(latitude) +','+ str(longitude) + ','+ speed
    sock.sendto(MESSAGE,(UDP_IP, UDP_PORT))
    logger.info(str(seqNo)+','+ speed + ',' + latitude + ',' + longitude)
    seqNo+=1

t = perpetualTimer(0.01,timerHandler) #10 hz
gpsThread = threading.Thread(target=readGPS)
gpsThread.start()
obdThread = threading.Thread(target=readOBD)
obdThread.start()
t.start()
