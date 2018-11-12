import csv, os, argparse
import numpy as np
def parse_args():
        parser = argparse.ArgumentParser("Post Processing")
        parser.add_argument("--csv-input-path", type=str, default="sensor_data.csv", help="Path of csv input file")
        parser.add_argument("--csv-output-path", type=str, default="training_data.csv", help="Path of csv output file")
        parser.add_argument("--time-horizon", type=int, default=200, help="window size")
        parser.add_argument("--mode", type=int, default=0, help="mode 0: single robot. mode 1: two robots")
        return parser.parse_args()

def post_process(arglist):
    input_data = []
    if arglist.mode == 0:
        offset = 0
    elif arglist.mode == 1:
        offset = 5

    with open(arglist.csv_input_path) as csv_input_file:
        with open(arglist.csv_output_path, "a") as csv_output_file:
            file_writer = csv.writer(csv_output_file, delimiter=',', quoting=csv.QUOTE_ALL)
            csv_reader = csv.reader(csv_input_file, delimiter=',')
            start = False
            for row in csv_reader:
                # move till you find the first row with CSI
                if start == False:
                    if row[11] == '0.0':
                        continue
                    start = True
                # copy the rest of the rows to input_data
                else:
                    input_data.append(row)
            # loop for all data
            for i, row in enumerate(input_data):
                # copy state info at current time
                out_data = row[1:(6+offset)]
                # calculate pdr over a time horizon
                pdr_count = 0.0
                if (len(input_data)-i) > arglist.time_horizon:
                    for j in range (i+1,i+arglist.time_horizon+1):
                        # copy future states info
                        for n in range (1,6+offset):
                            out_data.append(input_data[j][n])
                        if input_data[j][11] != '0.0':
                            pdr_count=pdr_count+1
                    for j in range (13, 193):
                        out_data.append(row[j])
                    out_data.append(pdr_count/arglist.time_horizon)
                    file_writer.writerow(out_data)






if __name__ == "__main__":
    arglist = parse_args()
    post_process(arglist)
