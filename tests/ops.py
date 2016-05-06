# -*- coding: utf-8 -*-


import os
import numpy as np
from .base import TestCase, tfdeploy as td
import tensorflow as tf


__all__ = ["OpsTestCase", "ScipyOpsTestCase", "TensorflowOpsTestCase"]


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

    def check(self, t, comp=None, ndigits=None, stats=False, abs=False, debug=False):
        rtf = t.eval(session=self.sess)
        rtd = td.Tensor(t, self.sess).eval()

        if hasattr(comp, "__call__"):
            return comp(rtf, rtd)

        if ndigits is None:
            ndigits = self.ndigits

        if debug:
            import pdb; pdb.set_trace()

        if isinstance(rtf, np.ndarray):
            self.assertEqual(rtf.dtype, rtd.dtype)
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

    def test_ops_have_tests(self):
        tests = [attr for attr in dir(self) if attr.startswith("test_")]
        for type in td.OperationRegister.classes:
            self.assertIn("test_" + type, tests)

    def test_Identity(self):
        t = tf.identity(self.random(3, 4))
        self.check(t)

    def test_Cast(self):
        t = tf.cast(self.random(3, 4).astype("float32"), tf.float64)
        self.check(t)

    def test_StringToNumber(self):
        t = tf.string_to_number(list("0123456789"))
        self.check(t)

    def test_Shape(self):
        t = tf.shape(self.random(3, 4, 5))
        self.check(t)

    def test_Size(self):
        t = tf.size(self.random(3, 4))
        self.check(t)

    def test_Rank(self):
        t = tf.rank(self.random(3, 3))
        self.check(t)

    def test_Reshape(self):
        t = tf.reshape(self.random(3, 4, 5), (2, -1))
        self.check(t)

    def test_Squeeze(self):
        t = tf.squeeze(self.random(1, 2, 1, 3, 3, 1))
        self.check(t)

    def test_ExpandDims(self):
        t = tf.expand_dims(self.random(2, 3, 3, 4), -2)
        self.check(t)

    def test_Slice(self):
        t = tf.slice(np.arange(3*4*8*6).reshape(3, 4, 8, 6), [1, 1, 2, 2], 4 * [2])
        self.check(t)

    def test_Split(self):
        for t in tf.split(1, 5, self.random(8, 50, 10, 2)):
            self.check(t)

    def test_Tile(self):
        t = tf.tile(self.random(3, 4, 5), [1, 2, 3])
        self.check(t)

    def test_Pad(self):
        t = tf.pad(self.random(3, 8, 5), [[1, 2], [2, 1], [1, 0]])
        self.check(t)

    def test_Concat(self):
        t = tf.concat(2, self.random((3, 4, 5), (3, 4, 5)))
        self.check(t)

    def test_Pack(self):
        t = tf.pack(self.random((3, 4, 5), (3, 4, 5)))
        self.check(t)

    def test_Unpack(self):
        for t in tf.unpack(self.random(6, 4, 5)):
            self.check(t)

    def test_ReverseSequence(self):
        x = self.random(3, 4, 10)
        t = tf.reverse_sequence(x, [5, 0, 0, 8], seq_dim=2, batch_dim=1)
        self.check(t)

    def test_Reverse(self):
        t = tf.reverse(self.random(3, 4, 10), [True, False, True])
        self.check(t)

    def test_Transpose(self):
        t = tf.transpose(self.random(4, 3, 5), perm=[2, 0, 1])
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

    def test_TrueDiv(self):
        t = tf.truediv(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_FloorDiv(self):
        t = tf.floordiv(*self.random((3, 5), (3, 5)))
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

    def test_Sum(self):
        t = tf.reduce_sum(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_Prod(self):
        t = tf.reduce_prod(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_Min(self):
        t = tf.reduce_min(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_Max(self):
        t = tf.reduce_max(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_Mean(self):
        t = tf.reduce_mean(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_All(self):
        t = tf.reduce_all(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_Any(self):
        t = tf.reduce_any(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)

    def test_SegmentSum(self):
        t = tf.segment_sum(self.random(4, 2, 3), np.array([0, 1, 1, 2]))
        self.check(t)

    def test_SegmentProd(self):
        t = tf.segment_prod(self.random(4, 2, 3), np.array([0, 1, 1, 2]))
        self.check(t)

    def test_SegmentMin(self):
        t = tf.segment_min(self.random(4, 2, 3), np.array([0, 1, 1, 2]))
        self.check(t)

    def test_SegmentMax(self):
        t = tf.segment_max(self.random(4, 2, 3), np.array([0, 1, 1, 2]))
        self.check(t)

    def test_SegmentMean(self):
        t = tf.segment_mean(self.random(4, 2, 3), np.array([0, 1, 1, 2]))
        self.check(t)

    def test_UnsortedSegmentSum(self):
        t = tf.unsorted_segment_sum(self.random(4, 2, 3), np.array([0, 2, 2, 1]), 3)
        self.check(t)

    def test_SparseSegmentSum(self):
        t = tf.sparse_segment_sum(self.random(4, 3, 2), [0, 2, 3], [0, 1, 1])
        self.check(t)

    def test_SparseSegmentMean(self):
        t = tf.sparse_segment_mean(self.random(4, 3, 2), [0, 2, 3], [0, 1, 1])
        self.check(t)

    def test_SparseSegmentSqrtN(self):
        t = tf.sparse_segment_sqrt_n(self.random(4, 3, 2), [0, 2, 3], [0, 1, 1])
        self.check(t)

    def test_ArgMin(self):
        t = tf.argmin(self.random(3, 4, 2), 1)
        self.check(t)

    def test_ArgMax(self):
        t = tf.argmax(self.random(3, 4, 2), 1)
        self.check(t)

    def test_ListDiff(self):
        l = np.random.randint(0, 5, 100)
        t1, t2 = tf.listdiff(l, l[::-2])
        self.check(t1)
        self.check(t2)

    def test_Where(self):
        t = tf.where([[True, False], [False, False], [True, False]])
        self.check(t)

    def test_Unique(self):
        t1, t2 = tf.unique([9, 3, 5, 7, 3, 9, 9])
        self.check(t1)
        self.check(t2)

    def test_InvertPermutation(self):
        t = tf.invert_permutation(np.random.permutation(10))
        self.check(t)

    def test_LinSpace(self):
        t = tf.linspace(0., 10., 15)
        self.check(t)

    def test_Range(self):
        t = tf.range(1, 10, 2)
        self.check(t)

    def test_RandomStandardNormal(self):
        t = tf.random_normal((40, 30), dtype="float32")
        # compare only dtype
        def comp(rtf, rtd):
            self.assertEqual(rtf.dtype, rtd.dtype)
        self.check(t, comp=comp)

    def test_TruncatedNormal(self):
        t = tf.truncated_normal((40, 300), dtype="float32")
        # compare dtype and 2-sigma truncation
        def comp(rtf, rtd):
            self.assertEqual(rtf.dtype, rtd.dtype)
            self.assertLessEqual(np.max(np.abs(rtd)), 2)
        self.check(t, comp=comp)

    def test_RandomUniform(self):
        t = tf.random_uniform((50, 80), -2, 3, dtype="float32")
        # compare only min, max and dtype
        def comp(rtf, rtd):
            self.assertLess(np.max(rtd), 3)
            self.assertGreaterEqual(np.min(rtd), -2)
            self.assertEqual(rtd.dtype, np.float32)
        self.check(t, comp=comp)

    def test_RandomUniformInt(self):
        # no python interface yet, but might be something like
        # t = tf.random_uniform_int((50, 80), -2, 3)
        # # compare only min and max
        # def comp(rtf, rtd):
        #     self.assertLess(np.max(rtd), 3)
        #     self.assertGreaterEqual(np.min(rtd), -2)
        # self.check(t, comp=comp)
        pass

    def test_RandomShuffle(self):
        t = tf.random_shuffle(self.random(10, 4))
        # compare only sum of first axis
        def comp(rtf, rtd):
            self.assertTrue(np.allclose(np.sum(rtf, axis=0), np.sum(rtd, axis=0)))
        self.check(t, comp=comp)

    def test_RandomCrop(self):
        t = tf.random_crop(self.random(3, 4, 8), [1, 2, 4])
        # compare only shape
        def comp(rtf, rtd):
            self.assertEqual(rtf.shape, rtd.shape)
        self.check(t, comp=comp)

    def test_Relu(self):
        t = tf.nn.relu(self.random(100) - 0.5)
        self.check(t)

    def test_Relu6(self):
        t = tf.nn.relu6((self.random(100) - 0.5) * 20)
        self.check(t)

    def test_Elu(self):
        t = tf.nn.elu(self.random(100) - 0.5)
        self.check(t)

    def test_Softplus(self):
        t = tf.nn.softplus(self.random(100) - 0.5)
        self.check(t)

    def test_Softsign(self):
        t = tf.nn.softsign(self.random(100) - 0.5)
        self.check(t)

    def test_BiasAdd(self):
        t = tf.nn.bias_add(*self.random((4, 5), (5,)))
        self.check(t)

    def test_Sigmoid(self):
        t = tf.nn.sigmoid(self.random(3, 4))
        self.check(t)

    def test_Tanh(self):
        t = tf.nn.tanh(self.random(3, 4))
        self.check(t)

    def test_Softmax(self):
        t = tf.nn.softmax(self.random(10, 5))
        self.check(t)


def factory(name, impl):
    classdict = {}

    def make_test_func(op):
        def test_func(self):
            op.use_impl(impl)
            getattr(OpsTestCase, test_name)(self)
        return test_func

    for op in td.Operation.__subclasses__():
        test_name = "test_" + op.__name__
        if impl in op.impls:
            test_func = make_test_func(op)
        else:
            test_func = None
        classdict[test_name] = test_func

    return OpsTestCase.__class__(name, (OpsTestCase,), classdict)


ScipyOpsTestCase = factory("ScipyOpsTestCase", td.IMPL_SCIPY)
TheanoOpsTestCase = factory("TheanoOpsTestCase", td.IMPL_THEANO)
TensorflowOpsTestCase = factory("TensorflowOpsTestCase", td.IMPL_TENSORFLOW)
