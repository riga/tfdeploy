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
    from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Reshape, \
        Conv2DTranspose, LSTM, LeakyReLU, Activation, RepeatVector, Lambda, LocallyConnected2D
    from keras.optimizers import Adam
    from keras.backend import tensorflow_backend as tfb
    import keras.backend as K
    from keras import applications as kapps  # for bigger prebuilt models

    KERAS_MISSING = False
except ImportError:
    KERAS_MISSING = True

UNSUPPORTED_LAYERS = ['Dropout', 'BatchNormalization', 'UpSampling2D', 'Convolution2DTranspose',
                      'LSTM', 'RepeatVector', 'LocallyConnected2D']

__all__ = ["KerasTestCase"]


@unittest.skipIf(KERAS_MISSING, "requires Keras to be installed")
class KerasTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super(KerasTestCase, self).__init__(*args, **kwargs)
        td.setup(tf)
        K.set_image_dim_ordering('tf')

    def test_deploy_tool(self):
        c_model = KerasTestCase._build_simple_2d(use_leakyrelu=True, use_pooling=True)
        t_model, in_mapping, out_mapping = td.deploy_keras(c_model)
        print(in_mapping)
        print(out_mapping)

        self.assertIsInstance(t_model, td.Model, "Output should be tfdeploy model")
        self.assertEqual(len(in_mapping), 1, "only one input")
        self.assertIn('Reshape_input', in_mapping, "Reshape not found in input")
        self.assertIn('MaxPooling2D', out_mapping, "MaxPooling not found in output")
        self.assertEqual(len(out_mapping), 1, "only one ouput")
        for c_mapping in [in_mapping, out_mapping]:
            for keras_name, tf_name in c_mapping.items():
                cur_tensor = t_model.get(tf_name)
                self.assertIsNotNone(cur_tensor, "Layer: {} -> TF:{}, not found in model".format(
                    keras_name, tf_name))
                self.assertIsInstance(cur_tensor, td.Tensor, "Layer should be tensor: {}".format(cur_tensor))

    def test_cnn_models(self):
        model_kwargs = dict(use_dense=False, use_dropout=False, use_pooling=False, use_bn=False, use_upsample=False,
                            use_conv2dtrans=False, use_lstm=False, use_leakyrelu=False, use_repeatvec=False,
                            use_lambda=False, use_locallyconnected=False)

        def _try_args(**kw_args):
            new_args = model_kwargs.copy()
            new_args.update(kw_args)
            return new_args

        test_models = [('base_cnn', KerasTestCase._build_simple_2d())]
        test_models += [(c_arg,
                         KerasTestCase._build_simple_2d(**_try_args(**{c_arg: True})))
                        for c_arg in model_kwargs.keys()]

        deployed_models = []
        for i, (model_name, cur_keras_model) in enumerate(test_models):
            model_layers = ','.join(map(lambda x: x.name, cur_keras_model.layers))
            out_path = "%04d.pkl" % i
            try:
                deployed_models += \
                    KerasTestCase.export_keras_model(cur_keras_model, out_path, model_name=model_layers)
            except td.UnknownOperationException as uoe:
                print('Model {}: {}'.format(i, model_name), 'could not be serialized', uoe)
                bad_layer_count = sum([us_layer in model_layers for us_layer in UNSUPPORTED_LAYERS])
                self.assertGreater(bad_layer_count, 0,
                                   "Model contains no unsupported layers {}, "
                                   "Unsupported Layers:{}".format(model_layers, UNSUPPORTED_LAYERS))

        self.assertGreater(len(deployed_models), 0, "No models could be tested")
        print("Testing #{} models".format(len(deployed_models)))
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
        kapp_kwargs = dict(
            input_shape=(99, 99, 3),
            weights=None,
            include_top=False  # so we can use different sizes
        )
        test_models = []

        test_models += [('Resnet50', kapps.ResNet50(**kapp_kwargs))]
        test_models += [('InceptionV3', kapps.InceptionV3(**kapp_kwargs))]
        test_models += [('VGG19', kapps.VGG19(**kapp_kwargs))]
        test_models += [('Xception', kapps.Xception(**kapp_kwargs))]

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
    def _build_simple_2d(use_dense=False, use_dropout=False, use_pooling=False, use_bn=False, use_upsample=False,
                         use_conv2dtrans=False, use_lstm=False, use_leakyrelu=False, use_repeatvec=False,
                         use_lambda=False, use_locallyconnected=False):
        """
        Simple function for building CNN models with various layers turned on and off
        :param use_dropout: 
        :param use_pooling: maxpooling2d
        :param use_bn: batchnormalization
        :param use_upsample: 
        :return: 
        """
        out_model = Sequential()
        if use_lstm:
            out_model.add(Reshape(target_shape=(1, 81), input_shape=(81,), name='Reshape_LSTM'))
            out_model.add(LSTM(81, name='LSTM'))
        if use_dense:
            out_model.add(Dense(81, input_shape=(81,), name='Dense'))
        if use_repeatvec:
            out_model.add(RepeatVector(3, input_shape=(81,), name='RepeatVector'))
            out_model.add(Lambda(lambda x: x[0, :], name='Lambda'))
        out_model.add(Reshape(target_shape=(9, 9, 1), input_shape=(81,), name='Reshape'))
        out_model.add(Convolution2D(2, (3, 3), input_shape=(9, 9, 1), name='Convolution2D'))
        if use_lambda:
            out_model.add(Lambda(lambda x: x + 1, name='Lambda_add'))
        if use_leakyrelu:
            out_model.add(LeakyReLU(0.1, name='LeakyRelu'))
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
        if use_locallyconnected:
            out_model.add(LocallyConnected2D(3, (3, 3), name='LocallyConnected2D'))

        KerasTestCase.compile_model(out_model)
        return out_model
