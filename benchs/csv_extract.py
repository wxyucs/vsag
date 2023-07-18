#!/usr/bin/env python3

import os
import csv
import h5py
import numpy as np
import subprocess

def cmd_get_csv():
    return input("1. csv filename: ")

def cmd_get_column():
    return input("2. select column: ")

def extract():
    csv_filename = cmd_get_csv()
    while not os.path.isfile(csv_filename):
        csv_filename = cmd_get_csv()
    line_count = int(subprocess.check_output(f'wc -l {csv_filename}', shell=True).split()[0]) - 1
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print(header)

        column_name = cmd_get_column()
        while column_name not in header:
            column_name = cmd_get_column()
        column_idx = header.index(column_name)

        hdf5_filename = input("3. output filename: ")
        dataset_name = input("4. dataset name: ")

        data = list()
        with h5py.File(hdf5_filename, 'a') as hdf5file:
            i = 0
            for row in reader:
                data.append([int(val) for val in row[column_idx].split(',')])

                i += 1
                if i % (line_count // 100) == 0:
                    print(f"\rcsv parsing ... {i / (line_count // 100)}%", end="")
            print("\nconverting ...")
            data = np.array(data)

            if dataset_name in hdf5file.keys():
                overwrite = input(f"{dataset_name} existed, overwrite?[y/N]")
                overwrite = overwrite.upper()
                if overwrite != "Y":
                    print("not overwrite, discarding data ...")
                    del data
                    return
                else:
                    del hdf5file[dataset_name]
            hdf5file.create_dataset(dataset_name, data.shape, data=data)
    print("extract done")

if __name__ == "__main__":
    while True:
        extract()

