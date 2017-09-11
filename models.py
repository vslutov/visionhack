from keras.applications.inception_v3 import InceptionV3
from keras.applications import VGG16
from keras.layers import Dense
from keras.models import Model

extractor = InceptionV3(include_top=False, pooling='avg')

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

base_model = VGG16()
pop_layer(base_model)
top = base_model.layers[-1].output
top = Dense(9, activation='sigmoid')(top)

framelabel_predictor = Model(base_model.input, top)
framelabel_predictor.summary()
