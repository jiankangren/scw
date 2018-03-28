#!/usr/bin/python
import obd, sys, subprocess, socket
from time import sleep

subprocess.Popen(["bash", "setNetwork.sh 2"])

UDP_IP = "192.168.1.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

#conn = obd.OBD()
#c = obd.commands[1][13]
while True:
   # r = conn.query(c)
   # print(r)
    data, addr = sock.recvfrom(1024)
    print (data)
    sleep(0.01)

