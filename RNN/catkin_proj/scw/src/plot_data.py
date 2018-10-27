import matlab.engine
import matplotlib.pyplot as plt
import io

eng = matlab.engine.start_matlab()
print("started matlab")
fh = io.open("out.fifo", "rb")
cur = 0
count = 1
ret = 0
fh.read(1)
i = 0

x_data = []
y_data = []
#fig = plt.gcf()
#fig.show()
#fig.canvas.draw()
while True:
    field_len = int.from_bytes(fh.read(1),byteorder='big')
    code = int.from_bytes(fh.read(1),byteorder='big')
   # code = fh.read(1)
    cur = cur+3
    if code == 187:
        bytes = fh.read(field_len-1)
        cur = cur + field_len - 1
        if len(bytes) != (field_len-1):
            continue
    if code == 187:
        csi_entry = eng.read_bf_file_realTime_python(bytes)
        csi = eng.get_scaled_csi(csi_entry)
        eff_SNR = eng.db(eng.get_eff_SNRs(csi), 'pow')
        print("eff SNR ", eff_SNR[0][3])
        fh.read(1)
       # x_data.append(i)
        #y_data.append(eff_SNR[0][3])
        #line.set_xdata(x_data)
        #line.set_ydata(y_data)
        #plt.scatter(i,float(eff_SNR[0][3]))
        #plt.show(block=False)

        #plt.scatter(i,int(eff_SNR[0][3]))
        #fig.canvas.draw()
        i = i +1

#    output = eng.snr_recorder_python(fh)
# read python file
# call read_bf 
# calculate esnr
# send data to server

#print("called script")
#print (output)
