
import datetime, csv
import argparse
import _thread as thread
import matlab.engine
import time, math, sys
#from tf.transformations import euler_from_quaternion
import io, os
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
        self.csi = None
        self.x = 0.0
        self.timestamp = ''
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.y = 0.0
        self.linear_x = 0.0
        #self.eng = matlab.engine.start_matlab()
        print("matlab started")
        #self.fh = io.open(arglist.fifo_path, "rb")
        self.fh = os.open(arglist.fifo_path, os.O_RDONLY)
        print("read file")


    def read_esnr(self, arglist):
        count = 1
        ret = 0
        #os.read(self.fh,1)
        #self.fh.read(1)
        file_out = open('log3.dat', 'wb')
        while True:
            try:
                file_data = os.read(self.fh,1)
                file_out.write(file_data)
                code = int.from_bytes(file_data ,byteorder='big')
                file_data = os.read(self.fh,1)
                file_out.write(file_data)
                field_len = int.from_bytes(file_data ,byteorder='big')
                if code == 193:
                    if field_len == 129:
                        bytes = os.read(self.fh,field_len-1)
                    #bytes = self.fh.read(field_len-1)
                        print(bytes[24:55])#.decode("utf-8")

                #field_len = int.from_bytes(file_data,byteorder='big')
                #field_len = int.from_bytes(self.fh.read(1), byteorder='big')
                #print("len ", field_len)
                #code = int(self.fh.read(1).encode('hex'),16)
                #file_data = os.read(self.fh,1)
                #file_out.write(file_data)
                #code = int.from_bytes(file_data  ,byteorder='big')
                #code = int.from_bytes(self.fh.read(1), byteorder='big')
                #print("code", code)
               #code = fh.read(1)
                #if code == 187:
                #    bytes = os.read(self.fh,field_len-1)
                #    file_out.write(bytes)
                    ##bytes = self.fh.read(field_len-1)
                #    if len(bytes) != (field_len-1):
                #        self.fh.close()
                #        break;
                  #  print(self.eng.triarea(2))
                    #print(bytes)
                    #csi_entry = self.eng.read_bf_file_realTime_python(bytes)
                    #self.csi = self.eng.get_scaled_csi(csi_entry)
                    # calculate snr just in case
                    ###self.eff_SNR = self.eng.db(self.eng.get_eff_SNRs(self.csi), 'pow')
                    #print("eff SNR ", self.eff_SNR)
                    #self.fh.read(1)
                #    os.read(self.fh,1)
                #elif code == 193:
                #    if field_len == 129:
                #        bytes = os.read(self.fh,field_len-1)
                #        file_out.write(bytes)
                        #bytes = self.fh.read(field_len-1)
                #        self.timestamp = (bytes[24:29])#.decode("utf-8")
            except KeyboardInterrupt:
                file_out.close()
                sys.exit()


    def test(self):
        while True:
            #print("eff SNR ", self.eff_SNR)
            #print("csi", self.csi)
            #print("timestamp", self.timestamp)
            time.sleep(1)
if __name__ == "__main__":
    arglist = parse_args()
    sr = SensorRecorder(arglist)
    thread.start_new_thread(sr.read_esnr, (arglist, ))
    sr.test()
