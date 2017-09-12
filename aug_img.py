import imgaug.augmenters as iaa
import argparse
import os
import cv2

def aug(images):
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-7, 7),
                   translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, mode='reflect'),
        iaa.GaussianBlur(sigma=(0.0, 0.7)),
        iaa.AdditiveGaussianNoise(scale=0.02 * 255),
        iaa.Multiply((0.7, 1.3)),
        iaa.Add((-20, 20), per_channel=0.3),
        iaa.ContrastNormalization((0.7, 1.3))
        ])

    return seq.augment_images(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="input dir with images")
    parser.add_argument("-o", type=str, required=True, help="ouput dir")

    args = parser.parse_args()
    indir = args.i
    output_dir = args.o

    images = []
    for img_name in os.listdir(indir):
        images.append(cv2.imread(os.path.join(indir, img_name)))

    aug_images = aug(images)

    for i, img in enumerate(aug_images):
        name = '{}.jpg'.format(i)
        cv2.imwrite(os.path.join(output_dir, name), img)


if __name__ == '__main__':
    main()




