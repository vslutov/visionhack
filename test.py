import cv2
import argparse
import os
import tqdm
import skvideo.io
import skimage.io
import skimage.transform
import numpy as np
from models import extractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help='input dir with video')
    parser.add_argument("-o", type=str, required=True, help='output dir')

    args = parser.parse_args()
    indir = args.i
    outdir = args.o

    video_names = sorted(os.listdir(indir))
    video_names = list(filter(lambda x: os.path.splitext(x)[1] == '.avi', video_names))
    for video_name in tqdm.tqdm(video_names):
        full_name = os.path.join(indir, video_name)

        reader = cv2.VideoCapture(full_name)
        num_frame = 0
        frames = []

        while True:
            ret, frame = reader.read()
            if ret:
                frame = cv2.resize(frame, dsize=(299, 299))
                frames.append(frame)
            else:
                break

        result = extractor.predict(np.array(frames[:302]), batch_size=64)
        out_path = os.path.join(outdir, video_name + ".npy")
        np.save(out_path, result.reshape((-1,)))

if __name__ == '__main__':
    main()

