import cv2
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten, Input


class DeepCalib:

    name = "DeepCalib"
    optimizer = keras.optimizers.Adam(learning_rate=(10 ** -6))
    loss = {'output_focal': 'logcosh', 'output_distortion': 'logcosh'}
    metrics = {'output_focal': 'logcosh', 'output_distortion': 'logcosh'}

    def __init__(self):

        self.input_shape = (299, 299, 3)

    def __call__(self):

        self.main_input = Input(shape=self.input_shape, dtype='float32', name='main_input')
        self.phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=self.main_input, input_shape=self.input_shape)
        phi_features = self.phi_model.output
        self.phi_flattened = Flatten(name='phi-flattened')(phi_features)
        final_output_focal = Dense(1, activation='sigmoid', name='output_focal')(self.phi_flattened)
        final_output_distortion = Dense(1, activation='sigmoid', name='output_distortion')(self.phi_flattened)

        #for layer in self.phi_model.layers:
            #layer.name = layer.name + '_phi'

        model = Model(
            name=self.name,
            inputs=self.main_input,
            outputs=[final_output_focal, final_output_distortion]
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            jit_compile=True
        )

        return model
