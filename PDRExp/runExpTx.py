#!/usr/bin/python
import obd, sys, subprocess, socket, serial
from time import sleep


subprocess.Popen( "./setNetwork.sh 1", shell=True)
sleep(1)

UDP_IP = "192.168.1.2"
UDP_PORT = 5005
MESSAGE = "NODATA"

ser = serial.Serial('/dev/ttyUSB0', 4800, timeout=1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#conn = obd.OBD()
#c = obd.commands[1][13]
while True:
    #r = conn.query(c)
    #print(r)
    line = ser.readline()
    if "GPGGA" in line:
        latitude = line[18:26]
        longitude = line[31:39]
        MESSAGE = str(longitude)+',' +str(latitude)
    else:
        MESSAGE = "NODATA"
    sock.sendto(MESSAGE,(UDP_IP, UDP_PORT))
    sleep(0.01)

