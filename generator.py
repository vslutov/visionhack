def custom_generator(path, batch_size):
    imgs_files = sorted(i for i in os.listdir(path) if i.endswith(".jpg"))
    with open(os.path.join(path, "y.txt")) as fin:
        y = [[float(i) for i in line.split()] for line in fin.readlines()]
    indices = list(range(len(y)))
    while True:
        shuffle(indices)
        for b_start in range(0, len(indices) // batch_size * batch_size, batch_size):
            imgs = [
                skimage.io.imread(os.path.join(path, imgs_files[indices[i]]))
                for i in range(b_start, min(len(indices), b_start + batch_size))
            ]
            sub_y = [
                y[indices[i]]
                for i in range(b_start, min(len(indices), b_start + batch_size))
            ]
            yield imgs, sub_y