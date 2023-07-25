#!/usr/bin/env python3

import os
import csv
import h5py
import numpy as np
import subprocess

def extract_from_csv(csvfile, line_count, id_column_idx, vector_column_idx, output_hdf5, dataset_name, overwrite):
    with h5py.File(output_hdf5, 'a') as hdf5file:
        with open(csvfile, 'r') as basefile:
            reader = csv.reader(basefile)
            header = next(reader)
            data = list()

            i = 0
            for row in reader:
                data.append([int(val) for val in row[vector_column_idx].split(',')])

                i += 1
                if i % (line_count // 100) == 0:
                    print(f"\r{dataset_name} parsing ... {i / (line_count // 100)}%", end="")
            print("\nconverting ...")
            data = np.array(data)

            if dataset_name in hdf5file.keys():
                if overwrite:
                    del hdf5file[dataset_name]
                else:
                    print(f"error: {dataset_name} exists in {output_hdf5}, overwrite is NOT allow")
                    exit(-1)
            hdf5file.create_dataset(dataset_name, data.shape, data=data)

def extract(base_csv, base_line_count, base_id_column_idx, base_vector_column_idx,
            query_csv, query_line_count, query_id_column_idx, query_vector_column_idx,
            output_hdf5, overwrite):
    extract_from_csv(base_csv, base_line_count, base_id_column_idx, base_vector_column_idx, output_hdf5, "base", overwrite)
    extract_from_csv(query_csv, query_line_count, query_id_column_idx, query_vector_column_idx, output_hdf5, "query", overwrite)
    
if __name__ == "__main__":
    # input of base dataset
    base_csv = ""
    while not os.path.isfile(base_csv):
        base_csv = input("1. base csv filename: ")
    base_line_count = int(subprocess.check_output(f'wc -l {base_csv}', shell=True).split()[0]) - 1
    with open(base_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print(header, "\n")

        base_id_column_name = None
        while base_id_column_name not in header:
            base_id_column_name = input("1.1 select id column: ")
        base_id_column_idx = header.index(base_id_column_name)

        base_vector_column_name = None
        while base_vector_column_name not in header:
            base_vector_column_name = input("1.2 select vector column: ")
        base_vector_column_idx = header.index(base_vector_column_name)

    # input of query dataset
    query_csv = ""
    while not os.path.isfile(query_csv):
        query_csv = input("2. query csv filename: ")
    query_line_count = int(subprocess.check_output(f'wc -l {query_csv}', shell=True).split()[0]) - 1
    with open(query_csv, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print(header, "\n")

        query_id_column_name = None
        while query_id_column_name not in header:
            query_id_column_name = input("2.1 select id column: ")
        query_id_column_idx = header.index(query_id_column_name)

        query_vector_column_name = None
        while query_vector_column_name not in header:
            query_vector_column_name = input("2.2 select vector column: ")
        query_vector_column_idx = header.index(query_vector_column_name)

    # input of output
    output_hdf5 = ""
    overwrite = True
    while output_hdf5 == "" or overwrite is False:
        output_hdf5 = input("3. output filename: ")
        output_exist = os.path.isfile(output_hdf5)
        if output_exist:
            overwrite = input(f"{output_hdf5} file existed, overwrite? [y/N]")
            overwrite = overwrite.upper() == "Y"
            if overwrite == "Y":
                print(f"output data would overwrite the base and query dataset in the output hdf5 file")

    plan_message = f"""
Extract Plan:
    base:
        row count: {base_line_count}
        id field name, colidx: {base_id_column_name}, {base_id_column_idx}
        vector field name, colidx: {base_vector_column_name}, {base_vector_column_idx}
    query:
        row count: {query_line_count}
        id field name, colidx: {query_id_column_name}, {query_id_column_idx}
        vector field name, colidx: {query_vector_column_name}, {query_vector_column_idx}
    output:
        filename: {output_hdf5}{", OVERWRITE" if output_exist else ""}
"""
    print(plan_message)

    extract(base_csv,
            base_line_count,
            base_id_column_idx,
            base_vector_column_idx,
            query_csv,
            query_line_count,
            query_id_column_idx,
            query_vector_column_idx,
            output_hdf5,
            overwrite)
