
import datetime, csv
import argparse
import _thread as thread
import matlab.engine
import time, math
#from tf.transformations import euler_from_quaternion
import io
def parse_args():
        parser = argparse.ArgumentParser("Sensor Recorder")
        parser.add_argument("--fifo-path", type=str, default="out.fifo", help="Path of wireless data fifo")
        parser.add_argument("--csv-path", type=str, default="sensor_data.csv", help="Path of csv output file")
        parser.add_argument("--object", action="store_true", default=False, help="track an object while the nodes are static")

        return parser.parse_args()
class SensorRecorder:
    def __init__(self, arglist):
        self.angular_z = 0.0
        self.yaw_angel = 0.0
        self.eff_SNR = [0]
        self.csi = [0]
        self.x = 0.0
        self.timestamp = ''
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.y = 0.0
        self.linear_x = 0.0
        self.eng = matlab.engine.start_matlab()
        print("matlab started")
        self.fh = io.open(arglist.fifo_path, "rb")

    def read_esnr(self):
        cur = 0
        count = 1
        ret = 0
        self.fh.read(1)
        while True:
            #field_len = int(self.fh.read(1).encode('hex'),16)
            field_len = int.from_bytes(self.fh.read(1), byteorder='big')
            #print("len ", field_len)
            #code = int(self.fh.read(1).encode('hex'),16)
            code = int.from_bytes(self.fh.read(1), byteorder='big')
            #print("code", code)
           #code = fh.read(1)
            cur = cur+3
            if code == 187:
                bytes = self.fh.read(field_len-1)
                cur = cur + field_len - 1
                if len(bytes) != (field_len-1):
                    self.fh.close()
                    break;
              #  print(self.eng.triarea(2))
                #print(bytes)
                csi_entry = self.eng.read_bf_file_realTime_python(bytes)
                self.csi = self.eng.get_scaled_csi(csi_entry)
                # calculate snr just in case
                self.eff_SNR = self.eng.db(self.eng.get_eff_SNRs(self.csi), 'pow')
                #print("eff SNR ", self.eff_SNR)
                self.fh.read(1)
            elif code == 193:
                bytes = self.fh.read(field_len)
                self.timestamp = str(bytes[18:25]) + str(bytes[0:5])

    def record(self, arglist):
        rate = rospy.Rate(100) # 10hz
        with open(arglist.csv_path,"a") as csvFile:
            Fileout = csv.writer(csvFile, delimiter=',', quoting=csv.QUOTE_ALL)
            while not rospy.is_shutdown():
                if arglist.object == True:
                    if abs(self.prev_y - self.y) > 0.001 and abs(self.prev_x - self.x) > 0.001:
                        self.linear_x = (self.x - self.prev_x) * 10
                        self.angular_z = (self.y - self.prev_y) * 10
                    self.prev_x = self.x
                    self.prev_y = self.y
                csvdata = [datetime.datetime.now(), self.x, self.y, self.linear_x, self.angular_z, self.yaw_angel, self.eff_SNR[0], self.csi[0], self.timestamp]
                print(csvdata)
                Fileout.writerow(csvdata)
                rate.sleep()
    def test(self):
        while True:
            print("eff SNR ", self.eff_SNR)
            time.sleep(0.1)
if __name__ == "__main__":
    arglist = parse_args()
    sr = SensorRecorder(arglist)
    thread.start_new_thread(sr.read_esnr, ())
    #sr.test()
    sr.record(arglist)
