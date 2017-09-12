#!/usr/bin/python3

import argparse
import os
import numpy as np
import tqdm
import skvideo.io
import skimage.io
import skimage.transform
import shutil
import cv2

labels = ["z", "c", "m", "t", "d", "b", "e", "x", "f"]
labels = {letter: labels.index(letter) for letter in labels}

def parse_mark(mark):
    label = mark[0]
    low, high = [int(i) for i in mark[1:].split("-")]
    return label, low, high

def parse_mark_line(line):
    splitted = line.split()
    return splitted[0], splitted[1:]

def create_classes(video_names, marks_files):
    global labels
    y = {video_name: [0] * 400 for video_name in video_names}
    to_remove = {video_name: set() for video_name in video_names}
    for mark_file in marks_files:
        with open(mark_file, "r") as fin:
            for line in fin:
                video_name, marks = parse_mark_line(line)
                for mark in marks:
                    label, low, high = parse_mark(mark)
                    if label != "u":
                        for i in range(low, high + 1):
                            print(i, video_name, mark_file)
                            y[video_name][i] |= (1 << labels[label])
                    else:
                        for i in range(low, high + 1):
                            to_remove[video_name].add(i)

    return y, to_remove

def save_file(outdir, filename, data):
    with open("{}/{}.txt".format(outdir, filename), "w") as fout:
        fout.write("\n".join([str(value) for value in data]))

def save_data(indir, outdir, y, ignore):
    classes_count = 2 ** len(labels)
    counter = 0
    for i in range(classes_count):
        out_video_dir = os.path.join(outdir, str(i))
        if os.path.exists(out_video_dir):
        	shutil.rmtree(out_video_dir)
        os.mkdir(out_video_dir)
    print("Old data removed")

    mapping = []
    new_y = []
    for video_name in tqdm.tqdm(sorted(y.keys())):
        full_name = os.path.join(indir, video_name)
        if not os.path.exists(full_name):
            continue
        reader = skvideo.io.FFmpegReader(full_name)

        for num_frame, frame in enumerate(reader.nextFrame()):
            if num_frame in ignore[video_name]:
                continue
            frame = skimage.transform.resize(frame, (299, 299))
            cl = y[video_name][num_frame]
            out_path = "{}/{}/{}.jpg".format(outdir, cl, counter)
            new_y.append(cl)
            counter += 1
            mapping.append("{}:{}".format(video_name, num_frame))
            skimage.io.imsave(out_path, frame)
        print("Video {} is ready!".format(video_name))

    save_file(outdir, "y", new_y)
    save_file(outdir, "mapping", mapping)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="input dir with video")
    parser.add_argument("-o", type=str, required=True, help="output dir")
    parser.add_argument("-m", type=str, required=True, help="dir with marks files")

    args = parser.parse_args()
    indir = args.i
    outdir = args.o

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    video_names = [
        name
        for name in sorted(os.listdir(indir))
        if os.path.splitext(name)[1] == ".avi"
    ]

    marks_names = [
        os.path.join(args.m, name)
        for name in sorted(os.listdir(args.m))
        if os.path.splitext(name)[1] == ".txt"
    ]

    y, ignore = create_classes(video_names, marks_names)
    print("Marks parsed")
    save_data(indir, outdir, y, ignore)

if __name__ == "__main__":
    main()
