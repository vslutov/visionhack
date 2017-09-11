import cv2
import argparse
import os
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help='input dir with video')
    parser.add_argument("-o", type=str, required=True, help='output dir')
    parser.add_argument('--scale', type=float, default=1, help='resize factor')

    args = parser.parse_args()
    indir = args.i
    outdir = args.o
    scale = args.scale

    if not os.path.exists(outdir):
        os.mkdir(outdir)


    video_names = os.listdir(indir)
    video_names = filter(lambda x: os.path.splitext(x)[1] == '.avi', video_names)
    for video_name in tqdm.tqdm(video_names):
        full_name = os.path.join(indir, video_name)
        out_video_dir = os.path.join(outdir, video_name)
        if not os.path.exists(out_video_dir):
            os.mkdir(out_video_dir)

        cap = cv2.VideoCapture(full_name)
        num_frame = 0

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                out_path = '{}/{}.jpg'.format(out_video_dir, num_frame)
                cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                num_frame += 1
            else:
                break

if __name__ == '__main__':
    main()
