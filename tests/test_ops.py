# -*- coding: utf-8 -*-


import numpy as np
from base import TestCase, tfdeploy as td


__all__ = ["OpsTestCase"]


class OpsTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(OpsTestCase, self).__init__(*args, **kwargs)

        self.a = np.arange(20).reshape((4, 5))
        self.b = np.arange(20, 40).reshape((4, 5))
        self.c = np.arange(20, 30).reshape((5, 2))
        self.d = np.arange(20, 30, 0.1).reshape(50, 2)
        self.e = np.arange(12).reshape((4, 3))
        self.f = np.arange(12, 24).reshape((4, 3))
        self.n = 3.5

    def compare_arrays(self, a, b):
        return self.assertTrue((a == b).all())

    def test_Identity(self):
        self.compare_arrays(td.Identity.func(self.a), self.a)

    def test_Add(self):
        self.assertEqual(td.Add.func(self.a, self.b).sum(), 780)

    def test_Sub(self):
        self.assertEqual(td.Sub.func(self.a, self.b).sum(), -400)

    def test_Mul(self):
        self.assertEqual(td.Mul.func(self.a, self.n).sum(), 665)

    def test_Div(self):
        self.assertEqual(round(td.Div.func(self.a, self.n).sum(), 5), 54.28571)

    def test_MatMul(self):
        self.assertEqual(td.MatMul.func(self.a, self.c).sum(), 9470)

    def test_Cross(self):
        print td.Cross.func(self.e, self.f)
        self.assertEqual(td.Cross.func(self.e, self.f).sum(), 0)

    def test_Round(self):
        self.assertEqual(td.Round.func(self.d).sum(), 2500)

    def test_Floor(self):
        self.assertEqual(td.Floor.func(self.d).sum(), 2450)

    def test_Ceil(self):
        self.assertEqual(td.Ceil.func(self.d).sum(), 2549)

    def test_Mod(self):
        self.assertEqual(td.Mod.func(self.a, self.b).sum(), 190)

    def test_Softmax(self):
        self.assertEqual(td.Softmax.func(self.a).sum(), 4)
