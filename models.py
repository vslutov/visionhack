from keras.applications.inception_v3 import InceptionV3

extractor = InceptionV3(include_top=False, pooling='max')
