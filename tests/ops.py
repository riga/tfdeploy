# -*- coding: utf-8 -*-


import os
import numpy as np
from .base import TestCase, td
import tensorflow as tf
from tensorflow.python.framework import device


__all__ = ["OpsTestCase"]


# get device from env
CPU, GPU = range(2)
DEVICE = CPU
if os.environ.get("TD_TEST_GPU", "").lower() in ("1", "yes", "true"):
    DEVICE = GPU
DEVICE_ID = "/%s:0" % ["cpu", "gpu"][DEVICE]

# setup td
td.setup(tf)

# optimize for scipy depending on env
if os.environ.get("TD_TEST_SCIPY", "").lower() in ("1", "yes", "true"):
    td.optimize(td.IMPL_SCIPY)


class OpsTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(OpsTestCase, self).__init__(*args, **kwargs)

        # add the device to the "_device_function_stack" of the default graph
        dev = device.merge_device(DEVICE_ID)
        tf.get_default_graph()._device_function_stack.append(dev)

        # create a tf session
        self.sess = tf.Session()

        self.ndigits = 7

    def check(self, t, comp=None, ndigits=None, stats=False, abs=False, debug=False):
        if td._tf_version[:3] < (0, 12, 0):
            self.sess.run(tf.initialize_all_variables())
        else:
            self.sess.run(tf.global_variables_initializer())

        if not isinstance(t, tuple):
            t = (t,)

        for _t in t:
            rtf = _t.eval(session=self.sess)
            rtd = td.Tensor(_t, self.sess).eval()

            if debug:
                import pdb; pdb.set_trace()

            if ndigits is None:
                ndigits = self.ndigits

            if hasattr(comp, "__call__"):
                return comp(rtf, rtd)

            if isinstance(rtf, np.ndarray):
                self.assertEqual(rtf.dtype, rtd.dtype)
                if abs:
                    rtf = np.abs(rtf)
                    rtd = np.abs(rtd)
                if not stats:
                    self.assertTrue(np.allclose(rtf, rtd, atol=0.1**ndigits))
                else:
                    self.assertEqual(round(rtf.sum(), ndigits), round(rtd.sum(), ndigits))
                    self.assertEqual(round(rtf.mean(), ndigits), round(rtd.mean(), ndigits))
            elif isinstance(rtf, float):
                self.assertEqual(round(rtf, ndigits), round(rtd, ndigits))
            else:
                self.assertEqual(rtf, rtd)

    def random(self, *shapes, **kwargs):
        if all(isinstance(i, int) for i in shapes):
            if kwargs.get("complex", False):
                return (self.random(*shapes) + 1j * self.random(*shapes)).astype(np.complex64)
            else:
                return np.random.rand(*shapes)
        else:
            return tuple(self.random(*shape) for shape in shapes)

    def test_ops_have_tests(self):
        tests = [attr for attr in dir(self) if attr.startswith("test_")]
        for type in td.OperationRegister.classes:
            self.assertIn("test_" + type, tests)


    #
    # sequences
    #

    def test_LinSpace(self):
        t = tf.linspace(0., 10., 15)
        self.check(t)

    def test_Range(self):
        t = tf.range(1, 10, 2)
        self.check(t)


    #
    # random tensors
    #

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

    def test_random_crop(self):
        t = tf.random_crop(self.random(3, 4, 8), [1, 2, 4])
        # compare only shape
        def comp(rtf, rtd):
            self.assertEqual(rtf.shape, rtd.shape)
        self.check(t, comp=comp)


    #
    # casting
    #

    def test_Cast(self):
        t = tf.cast(self.random(3, 4).astype("float32"), tf.float64)
        self.check(t)

    def test_StringToNumber(self):
        t = tf.string_to_number(list("0123456789"))
        self.check(t)


    #
    # shapes and shaping
    #

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


    #
    # slicing and joining
    #

    def test_Slice(self):
        t = tf.slice(np.arange(3*4*8*6).reshape(3, 4, 8, 6), [1, 1, 2, 2], 4 * [2])
        self.check(t)

    def test_Split(self):
        for t in tf.split(self.random(8, 50, 10, 2), 5, 2):
            self.check(t)

    def test_SplitV(self):
        for t in tf.split(self.random(8, 50, 10, 2), [10, 30, 5, 1, 4], 1):
            self.check(t)

    def test_Tile(self):
        t = tf.tile(self.random(3, 4, 5), [1, 2, 3])
        self.check(t)

    def test_Pad(self):
        t = tf.pad(self.random(3, 8, 5), [[1, 2], [2, 1], [1, 0]])
        self.check(t)

    def test_ConcatV2(self):
        aaa = self.random((3, 4, 5), (3, 4, 5))
        t = tf.concat(list(self.random((3, 4, 5), (3, 4, 5))), 2)
        self.check(t)

    def test_Pack(self):
        pass

    def test_Unpack(self):
        pass

    def test_Stack(self):
        t = tf.stack(list(self.random((3, 4, 5), (3, 4, 5))), 2)
        self.check(t)

    def test_Unstack(self):
        for t in tf.unstack(self.random(6, 4, 5), axis=1):
            self.check(t)

    def test_ReverseSequence(self):
        x = self.random(3, 4, 10)
        t = tf.reverse_sequence(x, [5, 0, 0, 8], seq_dim=2, batch_dim=1)
        self.check(t)

    def test_ReverseV2(self):
        t = tf.reverse(self.random(3, 4, 10), [1, 2])
        self.check(t)

    def test_Transpose(self):
        t = tf.transpose(self.random(4, 3, 5), perm=[2, 0, 1])
        self.check(t)


    #
    # arithmetic math ops
    #

    def test_Add(self):
        t = tf.add(*self.random((3, 4), (3, 4)))
        self.check(t)

    def test_Subtract(self):
        t = tf.subtract(*self.random((3, 4), (3, 4)))
        self.check(t)

    test_Sub = test_Subtract

    def test_Multiply(self):
        t = tf.multiply(*self.random((3, 5), (3, 5)))
        self.check(t)

    test_Mul = test_Multiply

    def test_scalar_mul(self):
        t = tf.scalar_mul(1, tf.Variable(self.random(3, 5)))
        self.check(t)

    def test_Div(self):
        t = tf.div(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_RealDiv(self):
        t = tf.div(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_TrueDiv(self):
        t = tf.truediv(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_FloorDiv(self):
        t = tf.floordiv(*self.random((3, 5), (3, 5)))
        self.check(t)

    def test_Mod(self):
        t = tf.mod(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_FloorMod(self):
        t = tf.floormod(*self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Cross(self):
        t = tf.cross(*self.random((4, 3), (4, 3)))
        self.check(t)


    #
    # basic math ops
    #

    def test_AddN(self):
        t = tf.add_n(self.random((4, 3), (4, 3)))
        self.check(t)

    def test_Abs(self):
        t = tf.abs(-self.random(4, 3))
        self.check(t)

    def test_Negative(self):
        t = tf.negative(self.random(4, 3))
        self.check(t)

    test_Neg = test_Negative

    def test_Sign(self):
        t = tf.sign(self.random(4, 3) - 0.5)
        self.check(t)

    def test_Inv(self):
        if td._tf_version[:2] <= (0, 11):
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

    def test_lbeta(self):
        t = tf.lbeta(self.random(4, 3))
        self.check(t)

    def test_Tan(self):
        t = tf.tan(self.random(4, 3))
        self.check(t)

    def test_Acos(self):
        t = tf.acos(self.random(4, 3))
        self.check(t)

    def test_Asin(self):
        t = tf.asin(self.random(4, 3))
        self.check(t)

    def test_Atan(self):
        t = tf.atan(self.random(4, 3))
        self.check(t)

    def test_Lgamma(self):
        t = tf.lgamma(self.random(4, 3))
        self.check(t)

    def test_Digamma(self):
        t = tf.digamma(self.random(4, 3))
        self.check(t)

    def test_Erf(self):
        t = tf.erf(self.random(4, 3))
        self.check(t)

    def test_Erfc(self):
        t = tf.erfc(self.random(4, 3))
        self.check(t)

    def test_SquaredDifference(self):
        t = tf.squared_difference(*self.random((3, 4, 4), (3, 4, 4)))
        self.check(t)

    def test_Igamma(self):
        t = tf.igamma(*self.random((3, 3), (3, 3)))
        self.check(t)

    def test_Igammac(self):
        t = tf.igammac(*self.random((3, 3), (3, 3)))
        self.check(t)

    def test_Zeta(self):
        t = tf.zeta(self.random(3, 3) + 2, self.random(3, 3))
        self.check(t)

    def test_Polygamma(self):
        t = tf.polygamma(np.array([1, 2, 3]).astype("float32"), np.array([4, 5, 6]).astype("float32"))
        self.check(t)

    def test_Betainc(self):
        t = tf.betainc(*self.random((3, 3), (3, 3), (3, 3)))
        self.check(t)


    #
    # matrix math ops
    #

    def test_Diag(self):
        t = tf.diag(self.random(3, 3))
        self.check(t)

    def test_DiagPart(self):
        t = tf.diag_part(self.random(3, 3))
        self.check(t)

    def test_MatrixDiagPart(self):
        if td._tf_version[:2] >= (0, 12):
            t = tf.matrix_diag_part(self.random(3, 4, 4, 5))
            self.check(t)

    def test_trace(self):
        t = tf.trace(self.random(3, 3))
        self.check(t)

    def test_MatMul(self):
        t = tf.matmul(*self.random((4, 3), (3, 5)), transpose_a=False, transpose_b=False)
        self.check(t)
        t = tf.matmul(*self.random((3, 4), (3, 5)), transpose_a=True, transpose_b=False)
        self.check(t)
        t = tf.matmul(*self.random((4, 3), (5, 3)), transpose_a=False, transpose_b=True)
        self.check(t)
        t = tf.matmul(*self.random((3, 4), (5, 3)), transpose_a=True, transpose_b=True)
        self.check(t)

    # def test_BatchMatMul(self):
    #     t = tf.batch_matmul(*self.random((2, 4, 4, 3), (2, 4, 3, 5)), adj_x=False, adj_y=False)
    #     self.check(t)
    #     t = tf.batch_matmul(*self.random((2, 4, 3, 4), (2, 4, 3, 5)), adj_x=True, adj_y=False)
    #     self.check(t)
    #     t = tf.batch_matmul(*self.random((2, 4, 4, 3), (2, 4, 5, 3)), adj_x=False, adj_y=True)
    #     self.check(t)
    #     t = tf.batch_matmul(*self.random((2, 4, 3, 4), (2, 4, 5, 3)), adj_x=True, adj_y=True)
    #     self.check(t)

    def test_MatrixDeterminant(self):
        t = tf.matrix_determinant(self.random(2, 3, 4, 3, 3))
        self.check(t)

    def test_MatrixInverse(self):
        t = tf.matrix_inverse(self.random(2, 3, 4, 3, 3), adjoint=False)
        self.check(t)
        t = tf.matrix_inverse(self.random(2, 3, 4, 3, 3), adjoint=True)
        self.check(t)

    def test_Cholesky(self):
        t = tf.cholesky(np.array(3 * [8, 3, 3, 8]).reshape(3, 2, 2).astype("float32"))
        self.check(t)

    def test_MatrixSolve(self):
        t = tf.matrix_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)), adjoint=False)
        self.check(t)
        t = tf.matrix_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)), adjoint=True)
        self.check(t)

    def test_MatrixTriangularSolve(self):
        t = tf.matrix_triangular_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)), adjoint=False, lower=False)
        self.check(t)
        t = tf.matrix_triangular_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)), adjoint=True, lower=False)
        self.check(t)
        t = tf.matrix_triangular_solve(*self.random((2, 3, 3, 3), (2, 3, 3, 1)), adjoint=False, lower=True)
        self.check(t)

    def test_MatrixSolveLs(self):
        t = tf.matrix_solve_ls(*self.random((2, 3, 3, 3), (2, 3, 3, 1)))
        self.check(t)

    def test_SelfAdjointEig(self):
        # legacy support
        pass

    def test_SelfAdjointEigV2(self):
        t = tf.self_adjoint_eig(np.array(3 * [3, 2, 2, 1]).reshape(3, 2, 2).astype("float32"))
        # the order of eigen vectors and values may differ between tf and np, so only compare sum
        # and mean
        # also, different numerical algorithms are used, so account for difference in precision by
        # comparing numbers with 4 digits
        self.check(t, ndigits=4, stats=True, abs=True)

    def test_Svd(self):
        t = tf.svd(self.random(4, 5, 3, 2).astype("float32"))
        self.check(t, ndigits=4, abs=True)


    #
    # complex number ops
    #

    def test_Complex(self):
        t = tf.complex(*self.random((3, 4), (3, 4)))
        self.check(t)

    def test_Conj(self):
        t = tf.conj(self.random(3, 4, complex=True))
        self.check(t)

    def test_Imag(self):
        t = tf.imag(tf.Variable(self.random(3, 4, complex=True)))
        self.check(t)

    def test_Real(self):
        t = tf.real(tf.Variable(self.random(3, 4, complex=True)))
        self.check(t)


    #
    # Fourier transform ops
    #

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

    def test_FFT3D(self):
        # only defined for gpu
        if DEVICE == GPU:
            t = tf.fft3d(self.random(3, 4, 5, complex=True))
            self.check(t)

    def test_IFFT3D(self):
        # only defined for gpu
        if DEVICE == GPU:
            t = tf.ifft3d(self.random(3, 4, 5, complex=True))
            self.check(t)


    #
    # reduction
    #

    def test_Sum(self):
        t = tf.reduce_sum(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        t = tf.reduce_sum(self.random(3, 4, 5), reduction_indices=(0, 1), keep_dims=True)
        self.check(t)
        t = tf.reduce_sum(self.random(3, 4, 5), reduction_indices=0, keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_sum(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)
            t = tf.reduce_sum(self.random(3, 4, 5), axis=(0, 1), keep_dims=True)
            self.check(t)
            t = tf.reduce_sum(self.random(3, 4, 5), axis=0, keep_dims=True)
            self.check(t)

    def test_Prod(self):
        t = tf.reduce_prod(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_prod(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)

    def test_Min(self):
        t = tf.reduce_min(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_min(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)

    def test_Max(self):
        t = tf.reduce_max(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_max(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)

    def test_Mean(self):
        t = tf.reduce_mean(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_mean(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)

    def test_All(self):
        t = tf.reduce_all(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_all(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)

    def test_Any(self):
        t = tf.reduce_any(self.random(3, 4, 5), reduction_indices=[0, 1], keep_dims=True)
        self.check(t)
        if td._tf_version[:3] >= (0, 12, 0):
            t = tf.reduce_any(self.random(3, 4, 5), axis=[0, 1], keep_dims=True)
            self.check(t)


    #
    # segmentation
    #

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


    #
    # sequence comparison and indexing
    #

    def test_ArgMin(self):
        t = tf.argmin(self.random(3, 4, 2), 1)
        self.check(t)

    def test_ArgMax(self):
        t = tf.argmax(self.random(3, 4, 2), 1)
        self.check(t)

    def test_ListDiff(self):
        if td._tf_version[:2] <= (0, 11):
            l = np.random.randint(0, 5, 100)
            t1, t2 = tf.listdiff(l, l[::-2])
            self.check(t1)
            self.check(t2)

    def test_Where(self):
        t = tf.where([[True, False], [False, False], [True, False]])
        self.check(t)

    def test_Unique(self):
        t = tf.unique([9, 3, 5, 7, 3, 9, 9], out_idx=tf.int32)
        self.check(t)

    def test_InvertPermutation(self):
        t = tf.invert_permutation(np.random.permutation(10))
        self.check(t)


    #
    # control flow ops
    #

    def test_Identity(self):
        t = tf.identity(self.random(3, 4))
        self.check(t)


    #
    # NN activation ops
    #

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


    #
    # NN convolution ops
    #

    def test_Conv1D(self):
        t = tf.nn.conv1d(np.arange(8000).reshape(1000, 2, 4).astype("float32"),
                         np.ones(80).reshape(2, 4, 10).astype("float32"),
                         1, "SAME")
        self.check(t)
        t = tf.nn.conv1d(np.arange(8000).reshape(1000, 2, 4).astype("float32"),
                         np.ones(80).reshape(2, 4, 10).astype("float32"),
                         2, "VALID")
        self.check(t)

    def test_Conv2D(self):
        t = tf.nn.conv2d(np.arange(24000).reshape(1000, 2, 3, 4).astype("float32"),
                         np.ones(160).reshape(2, 2, 4, 10).astype("float32"),
                         [1, 2, 3, 1], "SAME")
        self.check(t)
        t = tf.nn.conv2d(np.arange(24000).reshape(1000, 2, 3, 4).astype("float32"),
                         np.ones(160).reshape(2, 2, 4, 10).astype("float32"),
                         [1, 2, 5, 1], "VALID")
        self.check(t)

    def test_Conv3D(self):
        t = tf.nn.conv3d(np.arange(72000).reshape(1000, 2, 3, 3, 4).astype("float32"),
                         np.ones(320).reshape(2, 2, 2, 4, 10).astype("float32"),
                         [1, 1, 1, 1, 1], "SAME")
        self.check(t)
        t = tf.nn.conv3d(np.arange(72000).reshape(1000, 2, 3, 3, 4).astype("float32"),
                         np.ones(320).reshape(2, 2, 2, 4, 10).astype("float32"),
                         [1, 1, 1, 1, 1], "VALID")
        self.check(t)


    #
    # pooling ops
    #

    def test_AvgPool(self):
        t = tf.nn.avg_pool(np.arange(16).reshape(1, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        self.check(t)
        t = tf.nn.avg_pool(np.arange(16).reshape(1, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        self.check(t)

    def test_MaxPool(self):
        t = tf.nn.max_pool(np.arange(16).reshape(1, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        self.check(t)
        t = tf.nn.max_pool(np.arange(16).reshape(1, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
        self.check(t)
        t = tf.nn.max_pool(np.arange(64).reshape(2, 4, 8, 1).astype("float32"),
                           [1, 2, 2, 1], [1, 3, 2, 1], "VALID")
        self.check(t)

    def test_AvgPool3D(self):
        t = tf.nn.avg_pool3d(np.arange(64).reshape(1, 4, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], "SAME")
        self.check(t)
        t = tf.nn.avg_pool3d(np.arange(48).reshape(1, 4, 4, 3, 1).astype("float32"),
                           [1, 2, 2, 1, 1], [1, 2, 2, 1, 1], "VALID")
        self.check(t)

    def test_MaxPool3D(self):
        t = tf.nn.max_pool3d(np.arange(64).reshape(1, 4, 4, 4, 1).astype("float32"),
                           [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], "SAME")
        self.check(t)
        t = tf.nn.max_pool3d(np.arange(48).reshape(1, 4, 4, 3, 1).astype("float32"),
                           [1, 2, 2, 1, 1], [1, 2, 2, 1, 1], "VALID")
        self.check(t)
