# -*- coding: utf-8 -*-


import os
import numpy as np
from base import TestCase, tfdeploy as td
import tensorflow as tf


__all__ = ["OpsTestCase"]


# get device from env
CPU, GPU = range(2)
DEVICE = CPU
if os.environ.get("TD_TEST_GPU", "").lower() in ("1", "yes", "true"):
    DEVICE = GPU
DEVICE_ID = "/%s:0" % ["cpu", "gpu"][DEVICE]


class OpsTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(OpsTestCase, self).__init__(*args, **kwargs)

        # add the device to the "_device_function_stack" of the default graph
        dev = tf.python.framework.device.merge_device(DEVICE_ID)
        tf.get_default_graph()._device_function_stack.append(dev)

        # create a tf session
        self.sess = tf.Session()

        self.ndigits = 7

    def check(self, t, ndigits=None, stats=False, abs=False, debug=False):
        rtf = t.eval(session=self.sess)
        rtd = td.Tensor(t, self.sess).eval()

        if ndigits is None:
            ndigits = self.ndigits

        if debug:
            import pdb; pdb.set_trace()

        if isinstance(rtf, np.ndarray):
            if not stats:
                self.assertTrue(np.allclose(rtf, rtd))
            else:
                if abs:
                    rtf = np.abs(rtf)
                    rtd = np.abs(rtd)
                self.assertEqual(round(rtf.sum(), ndigits), round(rtd.sum(), ndigits))
                self.assertEqual(round(rtf.mean(), ndigits), round(rtd.mean(), ndigits))
        elif isinstance(rtf, float):
            self.assertEqual(round(rtf, ndigits), round(rtd, ndigits))
        else:
            self.assertEqual(rtf, rtd)

    def random(self, *shapes, **kwargs):
        if all(isinstance(i, int) for i in shapes):
            if kwargs.get("complex", False):
                return self.random(*shapes) + 1j * self.random(*shapes)
            else:
                return np.random.rand(*shapes)
        else:
            return tuple(self.random(*shape) for shape in shapes)

    def test_Identity(self):
        t = tf.identity(self.random(3, 4))
        self.check(t)

    def test_Add(self):
        t = tf.add(*self.random((3, 4), (3, 4)))
        self.check(t)

    def test_Sub(self):
        t = tf.sub(*self.random((3, 4), (3, 4)))
        self.check(t)

    def test_Mul(self):
        t = tf.mul(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_Div(self):
        t = tf.div(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_Cross(self):
        t = tf.cross(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Mod(self):
        t = tf.mod(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_AddN(self):
        t = tf.add_n(self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Abs(self):
        t = tf.abs(-self.random(4, 3))
        self.check(t)

    def test_Neg(self):
        t = tf.neg(self.random(4, 3))
        self.check(t)

    def test_Sign(self):
        t = tf.sign(self.random(4, 3) - 0.5)
        self.check(t)

    def test_Inv(self):
        t = tf.inv(self.random(4, 3))
        self.check(t)

    def test_Square(self):
        t = tf.square(self.random(4, 3))
        self.check(t)

    def test_Round(self):
        t = tf.round(self.random(4, 3) - 0.5)
        self.check(t)

    def test_Sqrt(self):
        t = tf.sqrt(self.random(4, 3))
        self.check(t)

    def test_Rsqrt(self):
        t = tf.rsqrt(self.random(4, 3))
        self.check(t)

    def test_Pow(self):
        t = tf.pow(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Exp(self):
        t = tf.exp(self.random(4, 3))
        self.check(t)

    def test_Log(self):
        t = tf.log(self.random(4, 3))
        self.check(t)

    def test_Ceil(self):
        t = tf.ceil(self.random(4, 3) - 0.5)
        self.check(t)

    def test_Floor(self):
        t = tf.floor(self.random(4, 3) - 0.5)
        self.check(t)

    def test_Maximum(self):
        t = tf.maximum(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Minimum(self):
        t = tf.minimum(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Cos(self):
        t = tf.cos(self.random(4, 3))
        self.check(t)

    def test_Sin(self):
        t = tf.sin(self.random(4, 3))
        self.check(t)

    def test_Lgamma(self):
        t = tf.lgamma(self.random(4, 3))
        self.check(t)

    def test_Erf(self):
        t = tf.erf(self.random(4, 3))
        self.check(t)

    def test_Erfc(self):
        t = tf.erfc(self.random(4, 3))
        self.check(t)

    def test_Diag(self):
        t = tf.diag(self.random(3, 3))
        self.check(t)

    def test_Transpose(self):
        t = tf.transpose(self.random(4, 3, 5), perm=[2, 0, 1])
        self.check(t)

    def test_MatMul(self):
        t = tf.matmul(*self.random((4, 3), (3, 5)))
        self.check(t)

    def test_BatchMatMul(self):
        t = tf.batch_matmul(*self.random((2, 4, 3, 4), (2, 4, 3, 5)), adj_x=True)
        self.check(t)

    def test_MatrixDeterminant(self):
        t = tf.matrix_determinant(self.random(3, 3))
        self.check(t)

    def test_BatchMatrixDeterminant(self):
        t = tf.batch_matrix_determinant(self.random(2, 3, 4, 3, 3))
        self.check(t)

    def test_MatrixInverse(self):
        t = tf.matrix_inverse(self.random(3, 3))
        self.check(t)

    def test_BatchMatrixInverse(self):
        t = tf.batch_matrix_inverse(self.random(2, 3, 4, 3, 3))
        self.check(t)

    def test_Cholesky(self):
        t = tf.cholesky(np.array([8, 3, 3, 8]).reshape(2, 2).astype("float32"))
        self.check(t)

    def test_BatchCholesky(self):
        t = tf.batch_cholesky(np.array(3 * [8, 3, 3, 8]).reshape(3, 2, 2).astype("float32"))
        self.check(t)

    def test_SelfAdjointEig(self):
        t = tf.self_adjoint_eig(np.array([3,2,1, 2,4,5, 1,5,6]).reshape(3, 3).astype("float32"))
        # the order of eigen vectors and values may differ between tf and np, so only compare sum
        # and mean
        # also, different numerical algorithms are used, so account for difference in precision by
        # comparing numbers with 4 digits
        self.check(t, ndigits=4, stats=True, abs=True)

    def test_BatchSelfAdjointEig(self):
        t = tf.batch_self_adjoint_eig(np.array(3 * [3, 2, 2, 1]).reshape(3, 2, 2).astype("float32"))
        self.check(t, ndigits=4, stats=True)

    def test_MatrixSolve(self):
        t = tf.matrix_solve(*self.random((3, 3), (3, 1)))
        self.check(t)

    def test_BatchMatrixSolve(self):
        t = tf.batch_matrix_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)))
        self.check(t)

    def test_MatrixSolveLs(self):
        t = tf.matrix_solve_ls(*self.random((3, 3), (3, 1)))
        self.check(t)

    def test_Complex(self):
        t = tf.complex(*self.random((3, 4), (3, 4)))
        self.check(t)

    def test_ComplexAbs(self):
        t = tf.complex_abs(self.random(3, 4, complex=True))
        self.check(t)

    def test_Conj(self):
        t = tf.conj(self.random(3, 4, complex=True))
        self.check(t)

    def test_Imag(self):
        t = tf.imag(self.random(3, 4, complex=True))
        self.check(t)

    def test_Real(self):
        t = tf.real(self.random(3, 4, complex=True))
        self.check(t)

    def test_FFT2D(self):
        # only defined for gpu
        if DEVICE == GPU:
            t = tf.fft2d(self.random(3, 4, complex=True))
            self.check(t)

    def test_IFFT2D(self):
        # only defined for gpu
        if DEVICE == GPU:
            t = tf.ifft2d(self.random(3, 4, complex=True))
            self.check(t)

    def test_Softmax(self):
        t = tf.nn.softmax(self.random(10, 5))
        self.check(t)

    def test_Rank(self):
        t = tf.rank(self.random(3, 3))
        self.check(t)

    def test_Range(self):
        t = tf.range(1, 10, 2)
        self.check(t)
