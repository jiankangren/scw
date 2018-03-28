#!/usr/bin/python
import obd, sys, subprocess, socket, serial
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
handler = logging.FileHandler('hello.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s , %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)


subprocess.Popen( "./setNetwork.sh 2", shell=True)
sleep(1)
UDP_IP = "192.168.1.2"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

packetCounter = 0
speed = 'NODATA'
leaderSpeed = 'NODATA'
#conn = obd.OBD()
t = perpetualTimer(0.1,timerHandler)

while True:
   # r = conn.query(c)
   # print(r)
    data, addr = sock.recvfrom(1024)
    if data:
        packetCounter+=1
    #speed = obd.commands[1][13]

def timeHandler():
    logger.info(speed, leaderSpeed, str(packetCounter)
    packetCounter = 0
   
 	
