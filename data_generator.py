import keras.preprocessing.image
import numpy as np
import os.path
from random import shuffle
import skimage.io

def convert_class(class_number, count_bit):
    class_number = int(class_number)
    class_number = bin(class_number)[2:]
    class_number = (count_bit - len(class_number)) * '0' + class_number
    return [int(i) for i in class_number]


class WrapperDirectoryIterator(keras.preprocessing.image.DirectoryIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        marks = list(self.class_indices.values())
        classes = list(self.class_indices.keys())
        classes = [convert_class(cl, 9) for cl in classes]
        self.invert_class_indices = dict(zip(marks, classes))

    def next(self, *args, **kwargs):
        X, y = super().next(*args, **kwargs)
        new_y = []
        for val in y.argmax(axis=1):
            new_y.append(self.invert_class_indices[val])
        return X, np.array(new_y)


class WrapperImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False):
        return WrapperDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)

def custom_generator(path, batch_size):
    imgs_files = sorted(i for i in os.listdir(path) if i.endswith(".jpg"))
    with open(os.path.join(path, "y.txt")) as fin:
        y = [[float(i) for i in line.split()] for line in fin.readlines()]
    indices = list(range(len(imgs_files)))
    while True:
        shuffle(indices)
        for b_start in range(0, len(indices) // batch_size * batch_size, batch_size):
            imgs = [
                skimage.io.imread(os.path.join(path, imgs_files[indices[i]]))
                for i in range(b_start, min(len(indices), b_start + batch_size))
            ]
            sub_y = [
                y[int(imgs_files[indices[i]].split('.')[0])]
                for i in range(b_start, min(len(indices), b_start + batch_size))
            ]
            yield np.array(imgs)[:, ::2, ::2, :], np.array(sub_y)