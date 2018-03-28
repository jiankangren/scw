#!/usr/bin/python
import obd
from time import sleep

conn = obd.OBD()
c = obd.commands[1][13]
while True:
    r = conn.query(c)
    print(r)
    sleep(0.01)


