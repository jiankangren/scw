import csv, os, argparse
import numpy as np
def parse_args():
        parser = argparse.ArgumentParser("Post Processing")
        parser.add_argument("--csv-input-path1", type=str, default="../dataset/exp4_train_post-20.csv", help="Path of csv input file")
        parser.add_argument("--csv-input-path2", type=str, default="../dataset/exp4_test_post-20.csv", help="Path of csv input file")
        parser.add_argument("--csv-output-path", type=str, default="../dataset/exp4_post-20.csv", help="Path of csv output file")
        parser.add_argument("--time-horizon", type=int, default=200, help="window size")
        parser.add_argument("--mode", type=int, default=0, help="mode 0: single robot. mode 1: two robots")
        return parser.parse_args()

def post_process(arglist):
    input_data = []
    with open(arglist.csv_input_path1) as csv_input_file:
        with open(arglist.csv_input_path2) as csv_input_file2:
            with open(arglist.csv_output_path, "a") as csv_output_file:
                file_writer = csv.writer(csv_output_file, delimiter=',', quoting=csv.QUOTE_ALL)
                csv_reader = csv.reader(csv_input_file, delimiter=',')
                for row in csv_reader:
                    input_data.append(row)
                csv_reader = csv.reader(csv_input_file2, delimiter=',')
                for row in csv_reader:
                    input_data.append(row)
                # loop for all data
                for i, row in enumerate(input_data):
                    file_writer.writerow(row)

if __name__ == "__main__":
    arglist = parse_args()
    post_process(arglist)
