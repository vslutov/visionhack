import cv2
import argparse
import os
import tqdm
import skvideo.io
import skimage.io
import skimage.transform
import numpy as np
from keras.applications.inception_v3 import preprocess_input

from models import extractor

def predict_labels(video_path, extractor, img_shape, batch_size=16):
    reader = skvideo.io.FFmpegReader(video_path)
    num_frame = 0
    frames = []

    for frame in reader.nextFrame():
        frame = cv2.resize(frame, dsize=img_shape[::-1])
        frames.append(frame)

    result = extractor.predict(preprocess_input(np.array(frames[:302]).astype(float)), batch_size=batch_size)
    return result

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
        result = predict_labels(full_name, extractor, (299, 299), batch_size=64)
        out_path = os.path.join(outdir, video_name + ".npy")
        np.save(out_path, result)

if __name__ == '__main__':
    main()

