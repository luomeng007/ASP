# -*- coding:utf-8 -*-
"""
author: 15025
age: 26
e-mail: 1502506285@qq.com
time: 2020/12/1 10:07
software: PyCharm

Description:
    COPH: the abbreviation of Computational Photonics
    we should understand the structure of brag mirror
    it is a set of mirrors with two different refractive index appear alternately.
    and start with layer 1 and end with layer 2
    n1, n2: float or complex, Refractive indices of the layers of one period
    d1, d2: float, Thicknesses of layers of one period
    N: number of periods.
    periods: one layer with n1 and and another layer with n2. Both of them compose one period

"""

import numpy as np


class COPH:
    @staticmethod
    def bragMirror(n1, n2, d1, d2, N):
        # we are not sure the type of n1 and n2
        # so first we need to get their common_type
        # np.common_type function can only deal with ndarray, so we need to transfer n1, n2 first
        # default type after ndarray transfer for int is int32
        # default type after ndarray transfer for float is float64
        common_data_type = np.common_type(np.array(n1), np.array(n2))
        # we have N periods, so we have 2N layers, we create its corresponding indices
        epsilon = np.zeros(2 * N, dtype=common_data_type)
        # set epsilon value at odd position
        epsilon[::2] = n1 ** 2
        # set epsilon value at even position
        epsilon[1::2] = n2 ** 2
        # use same method to set thickness
        thickness = np.zeros(2 * N)
        thickness[::2] = d1
        thickness[1::2] = d2

        # return value for later use
        return epsilon, thickness


if __name__ == '__main__':
    main = COPH()
    epsilon_, thickness_ = main.bragMirror(np.sqrt(2.25), np.sqrt(15.21), 0.13, 0.05, 5)
    print("The array of epsilon_ is:")
    print(epsilon_)
    print("The array of thickness_ is:")
    print(thickness_)
