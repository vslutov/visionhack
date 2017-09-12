#!/usr/bin/python3

import argparse
import os
import numpy as np
import shutil
from random import shuffle
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="input dir with images")
    parser.add_argument("-t", type=str, required=True, help="train dir")
    parser.add_argument("-v", type=str, required=True, help="validation dir")
    parser.add_argument("-m", type=str, required=True, help="file with mapping")
    parser.add_argument("-s", type=float, required=True, help="Train set size (float from 0 to 1)")

    args = parser.parse_args()
    indir = args.i
    train_dir = args.t
    validation_dir = args.v

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)

    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(validation_dir)

    for i in range(512):
        os.mkdir(os.path.join(train_dir, str(i)))
        os.mkdir(os.path.join(validation_dir, str(i)))

    with open(args.m, "r") as fin:
        mapping = [m.split(":")[0] for m in fin.readlines()]

    video_names = list(set(mapping))
    shuffle(video_names)
    threshold = int(len(video_names) * args.s)
    train_videos = set(video_names[:threshold])

    for i in range(512):
        path = os.path.join(indir, str(i))
        for f in os.listdir(path):
            if not f.endswith(".jpg"):
                continue
            if mapping[int(f.split(".")[0])] in train_videos:
                subprocess.run([
                    "ln", 
                    os.path.join(path, f),
                    os.path.join(train_dir, str(i), f)
                ])
            else:
                subprocess.run([
                    "ln", 
                    os.path.join(path, f),
                    os.path.join(validation_dir, str(i), f)
                ])

if __name__ == "__main__":
    main()
