#!/usr/bin/python
import sys, subprocess, socket, serial, logging#,obd
import time
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
handler = logging.FileHandler('rx.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s , %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


def readOBD():
    global speed
    #speed = conn.query(c)

def readGPS():
    global latitude
    global longitude
    line = ser.readline()
    if "GPGGA" in line:
        latitude = line[18:26]
        longitude = line[31:39]
    else:
	    latitude = "NODATA"
        longitude = "NODATA"

subprocess.Popen( "./setNetwork.sh 2", shell=True)
sleep(1)
UDP_IP = "192.168.1.2"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

speed = 'NODATA'
leaderSpeed = 'NODATA'
latitude = 'NODATA'
longitude = 'NODATA'
#conn = obd.OBD()
#r = conn.query(c)
t = perpetualTimer(0.1,timerHandler)
gpsThread = threading.Thread(target=readGPS)
gpsThread.start()
obdThread = threading.Thread(target=readOBD)
obdThread.start()
t.start()
seqNo = -1
while True:
    global speed, latitude, longitude
    data, addr = sock.recvfrom(1024)
    dataArr = data.split(',')
    if seqNo != int(dataArr[0]):
        seqNo = int(dataArr[0])
        logger.info(str(seqNo) + ',' + speed + latitude + ',' + longitude + ',' + dataArr[1] + ',' + dataArr[2] + ',' + dataArr[3])
    
#def timeHandler():
#    logger.info(speed, leaderSpeed, str(packetCounter)
#    packetCounter = 0

 	
