# -*- coding: utf-8 -*-


import os
import unittest
from itertools import product

import numpy as np
import tensorflow as tf

from .base import TestCase, td

# noinspection PyUnresolvedReferences
try:

    from keras.models import Sequential, Model, Input
    from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Reshape, \
        Conv2DTranspose
    from keras.optimizers import Adam
    from keras.backend import tensorflow_backend as tfb
    import keras.backend as K
    from keras import applications as kapps  # for bigger prebuilt models

    KERAS_MISSING = False
except ImportError:
    KERAS_MISSING = True

UNSUPPORTED_LAYERS = ['Dropout', 'BatchNormalization', 'UpSampling2D', 'Convolution2DTranspose']

__all__ = ["KerasTestCase"]


@unittest.skipIf(KERAS_MISSING, "requires Keras to be installed")
class KerasTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super(KerasTestCase, self).__init__(*args, **kwargs)
        td.setup(tf)
        K.set_image_dim_ordering('tf')

    def test_simple_models(self):
        test_models = [KerasTestCase._build_simple_cnn(*iv) for iv in product(*([[True, False]] * 5))]
        deployed_models = []
        for i, cur_keras_model in enumerate(test_models):

            model_layers = ','.join(map(lambda x: x.name, cur_keras_model.layers))
            out_path = "%04d.pkl" % i
            try:
                deployed_models += \
                    KerasTestCase.export_keras_model(cur_keras_model, out_path, model_name=model_layers)
            except td.UnknownOperationException as uoe:
                print('Model {}: {}'.format(i, model_layers), 'could not be serialized', uoe)
                bad_layer_count = sum([us_layer in model_layers for us_layer in UNSUPPORTED_LAYERS])
                self.assertGreater(bad_layer_count, 0,
                                   "Model contains no unsupported layers {}, "
                                   "Unsupported Layers:{}".format(model_layers, UNSUPPORTED_LAYERS))

        for c_model_pkl in deployed_models:
            result = KerasTestCase.deploy_model(c_model_pkl)
            self.assertIsNotNone(result, "Result should not be empty")
            self.assertEqual(len(result.shape), 4, "Output should be 4D Tensor: {}".format(result.shape))
            os.remove(c_model_pkl['path'])

    @unittest.skip("Takes quite awhile to run (and fails for all models)")
    def test_big_models(self):
        """
        A test for bigger commonly used pretrained models (for this we skip the weights)
        :return: 
        """
        test_models = []
        test_models += [('Resnet50', kapps.ResNet50(weights=None))]
        test_models += [('InceptionV3', kapps.InceptionV3(weights=None))]
        test_models += [('VGG19', kapps.VGG19(weights=None))]

        for i, (model_name, cur_keras_model) in enumerate(test_models):

            model_layers = ','.join(map(lambda x: x.name, cur_keras_model.layers))
            out_path = "%04d.pkl" % i
            try:
                c_model_pkl = KerasTestCase.export_keras_model(cur_keras_model, out_path, model_name=model_layers)
            except td.UnknownOperationException as uoe:
                print('Model {}: {}'.format(i, model_layers), 'could not be serialized', uoe)
                bad_layer_count = sum([us_layer in model_layers for us_layer in UNSUPPORTED_LAYERS])
                self.assertGreater(bad_layer_count, 0,
                                   "Model contains no unsupported layers {}, "
                                   "Unsupported Layers:{}".format(model_layers, UNSUPPORTED_LAYERS))
                continue
            except tf.errors.RESOURCE_EXHAUSTED:
                # many of the bigger models take up quite a bit of GPU memory
                print('Model {} with #{} layers is too big for memory'.format(model_name, len(cur_keras_model.layers)))

            result = KerasTestCase.deploy_model(c_model_pkl, np.random.uniform(0, 1, size=(299, 299, 3)))
            self.assertIsNotNone(result, "Result should not be empty")
            self.assertEqual(len(result.shape), 4, "Output should be 4D Tensor: {}".format(result.shape))
            os.remove(c_model_pkl['path'])

    @staticmethod
    def deploy_model(c_model_pkl, input=None):
        model = td.Model(c_model_pkl['path'])
        inp, outp = model.get(c_model_pkl['input'], c_model_pkl['output'])
        if input is None:
            input = np.random.rand(50, 81)
        return outp.eval({inp: input})

    @staticmethod
    def export_keras_model(in_ks_model, out_path, model_name):
        td_model = td.Model()
        td_model.add(in_ks_model.get_output_at(0),
                     tfb.get_session())  # y and all its ops and related tensors are added recursively

        td_model.save(out_path)
        return [dict(path=out_path,
                     output=in_ks_model.get_output_at(0).name,
                     input=in_ks_model.get_input_at(0).name,
                     name=model_name)]

    @staticmethod
    def compile_model(i_model):
        i_model.compile(optimizer=Adam(lr=2e-3), loss='mse')

    @staticmethod
    def _build_simple_cnn(use_dropout=False, use_pooling=False, use_bn=False, use_upsample=False,
                          use_conv2dtrans=False):
        """
        Simple function for building CNN models with various layers turned on and off
        :param use_dropout: 
        :param use_pooling: maxpooling2d
        :param use_bn: batchnormalization
        :param use_upsample: 
        :return: 
        """
        out_model = Sequential()
        out_model.add(Reshape(target_shape=(9, 9, 1), input_shape=(81,), name='Reshape'))
        out_model.add(Convolution2D(2, (3, 3), input_shape=(9, 9, 1), name='Convolution2D'))
        if use_dropout:
            out_model.add(Dropout(0.5, name='Dropout'))
        if use_pooling:
            out_model.add(MaxPooling2D((2, 2), name='MaxPooling2D'))
        if use_upsample:
            out_model.add(UpSampling2D((2, 2), name='UpSampling2D'))
        if use_bn:
            out_model.add(BatchNormalization(name='BatchNormalization'))
        if use_conv2dtrans:
            out_model.add(Conv2DTranspose(2, kernel_size=(3, 3), strides=(2, 2), name='Convolution2DTranspose'))

        KerasTestCase.compile_model(out_model)
        return out_model
