#!/bin/bash
if ! iwconfig wlan0 | grep scw; then
sudo ifconfig wlan0 down
sudo iwconfig wlan0 mode Ad-hoc essid scw
sudo ifconfig wlan0 192.168.1.$1 netmask 255.255.255.0
sudo ifconfig wlan0 up
fi
