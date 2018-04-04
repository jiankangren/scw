#!/usr/bin/python
import sys, subprocess, socket, serial, logging,obd
import time, threading
import numpy as np
index = 0
sec = np.array([12500])
seqNo = np.array([12500])
pdr = np.array([12500])
currentSec = 36
firstSec = 0
lastSec = 0
sec[0] = 36
counter = 0
with open("./rx1.log" ,'r') as f:
    for line in f:
    	data = line.split(';')
	if int(str(str(data[0].split(',')[0]).split(' ')[1]).split(':')[2]) == currentSec:
		firstSec = int(data[0].split(',')[1])
		counter =
	else:
		pdr[index] = int(data[0].split(',')[1]) - firstSec
	sec[index] = 
	mins[index] = str(str(data[0].split(',')[0]).split(' ')[1]).split(':')[1]
	seqNo[index] = data[0].split(',')[1]
	

