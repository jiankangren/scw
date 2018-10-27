import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import datetime, csv
import argparse
import _thread as thread
import matlab.engine
import time, math
import numpy as np
#from tf.transformations import euler_from_quaternion
import io, os, sys
def parse_args():
        parser = argparse.ArgumentParser("Sensor Recorder")
        parser.add_argument("--fifo-path", type=str, default="out.fifo", help="Path of wireless data fifo")
        parser.add_argument("--csv-path", type=str, default="sensor_data.csv", help="Path of csv output file")
        #parser.add_argument("--object", action="store_true", default=False, help="track an object while the nodes are static")

        return parser.parse_args()
class SensorRecorder:
    def __init__(self, arglist):
        self.angular_z1 = 0.0
        self.angular_z2 = 0.0
        self.yaw_angel1 = 0.0
        self.yaw_angel2 = 0.0
        self.eff_SNR = [0]
        self.csi = np.zeros(180)
        self.x1 = 0.0
        self.x2 = 0.0
        self.timestamp = ''
        self.y1 = 0.0
        self.y2 = 0.0
        self.linear_x1 = 0.0
        self.linear_x2 = 0.0
        self.new_data = False
        self.eng = matlab.engine.start_matlab()
        print("matlab started")

        self.fh = os.open(arglist.fifo_path, os.O_RDONLY)
        print("read file")
        rospy.init_node('node_name')
        rospy.Subscriber("/turtlebot1/odom", Odometry, self.odom_callback1)
        rospy.Subscriber("/turtlebot2/odom", Odometry, self.odom_callback2)
        rospy.Subscriber("/turtlebot1/Robot_1/pose", PoseStamped, self.pose_callback1)
        rospy.Subscriber("/turtlebot1/Robot_2/pose", PoseStamped, self.pose_callback2)
    def odom_callback1(self,data):
        #rospy.loginfo(data.twist.twist.angular.z)
        self.angular_z1 = data.twist.twist.angular.z
        self.linear_x1 = data.twist.twist.linear.x

    def odom_callback2(self,data):
        #rospy.loginfo(data.twist.twist.angular.z)
        self.angular_z2 = data.twist.twist.angular.z
        self.linear_x2 = data.twist.twist.linear.x
       
    def pose_callback1(self,data):
        #rospy.loginfo(data.pose.orientation.z)
        self.yaw_angel1 = data.pose.orientation.z
        self.x1 =  data.pose.position.x
        self.y1 = data.pose.position.y
    def pose_callback2(self,data):
        #rospy.loginfo(data.pose.orientation.z)
        self.yaw_angel2 = data.pose.orientation.z
        self.x2 =  data.pose.position.x
        self.y2 = data.pose.position.y

    def record(self, arglist):
        rate = rospy.Rate(50) # 10Hz
        with open(arglist.csv_path,"a") as csvFile:
            Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
            while not rospy.is_shutdown():
                csvdata = [datetime.datetime.now(), self.x1, self.y1, self.linear_x1, self.angular_z1, self.yaw_angel1, self.x2, self.y2, self.linear_x2, self.angular_z2, self.yaw_angel2]
                if self.new_data == True:
                    csvdata.append(self.timestamp)
                    csvdata.append(self.eff_SNR[0])
                    for i in range(0,90):
                        csvdata.append(self.csi._real[i])
                        csvdata.append(self.csi._imag[i])
                    self.new_data = False
                else:
                    for i in range(0,182):
                        csvdata.append(0.0)
                #print(csvdata)
                Fileout.writerow(csvdata)
                rate.sleep()
    def read_esnr(self, arglist):
        while True:
            field_len = int.from_bytes(os.read(self.fh,2)  ,byteorder='big')
            print("field len", field_len)
            code = int.from_bytes(os.read(self.fh,1)  ,byteorder='big')
            print("code", code)
            if code == 187:
                bytes = os.read(self.fh,field_len-1)
                if len(bytes) != (field_len-1):
                    os.close(self.fh);
                    print("wrong length field")
                    sys.exit()
                csi_entry = self.eng.read_bf_file_realTime_python(bytes)
                self.csi = self.eng.get_scaled_csi(csi_entry)
                # calculate snr just in case
                self.eff_SNR = self.eng.db(self.eng.get_eff_SNRs(self.csi), 'pow')
                #print("eff SNR ", self.eff_SNR)
                self.new_data  = True
            elif code == 193:
                #if field_len == 129:
                bytes = os.read(self.fh,field_len-1)
                #bytes = self.fh.read(field_len-1)
                self.timestamp = (bytes[24:55]).decode("utf-8")

    def test(self):
        while True:
            time.sleep(1)
if __name__ == "__main__":
    arglist = parse_args()
    sr = SensorRecorder(arglist)
    thread.start_new_thread(sr.read_esnr, (arglist, ))
    #sr.test()
    sr.record(arglist)
