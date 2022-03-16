'''
cfg-size.py

Records the number of basic blocks in the main function of each binary.

Before running this script, install angr's dependencies with:
$ sudo apt install python3-dev libffi-dev build-essential virtualenvwrapper
Then install with:
$ pip3 install angr
'''

import sys
import os
import angr
import csv
import networkx as nx
import json
import gc
import random
import itertools
import multiprocessing as mp
import numpy as np
from io import StringIO


'''
Yield successive n-sized chunks from lst.
Source: https://stackoverflow.com/questions
/312443/how-do-you-split-a-list-into-evenly-sized-chunks
'''
def chunks(lst, n):

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


'''
This function runs in parallel to extract a binary's CFG and record
the number of basic blocks in the CFG of the main function.
'''
def worker(basenames, outfile, num):
    outfile = open(str(num) + "-" + outfile, 'a')

    for bname in basenames:
        path = os.path.join(sys.argv[1], bname)
        try:
            # Lift CFG from assembly
            p = angr.Project(path, load_options={"auto_load_libs": False})
            cfg = p.analyses.CFGFast()
            # normalize() ensures that basic blocks do not overlap
            cfg.normalize()

            # Record size (basic block count) of main
            outfile.write(f"{bname} {len(cfg.functions["main"].graph)}\n")

            # Destroy CFG immediately to keep memory use low
            del p
            gc.collect()
        
        except Exception as e:
            print(e)
            print(f"Skipping {bname} due to angr parsing error")
            del p
            gc.collect()
            continue

    outfile.close()


'''
Bootstraps parallel CFG extraction.
'''
def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <binary directory> <outfile> <# of parallel jobs>")
        exit(1)

    assert(os.path.isdir(sys.argv[1]))
    basenames = os.listdir(sys.argv[1])

    NUM_CORES = sys.argv[3]
    split_basenames = list(chunks(basenames, len(basenames) // NUM_CORES + 1))

    processes = [mp.Process(target=worker,
                            args=(split_basenames[i], sys.argv[2], i))
                            for i in range(len(split_basenames))]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
