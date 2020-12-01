# -*- coding:utf-8 -*-
"""
author: 15025
age: 26
e-mail: 1502506285@qq.com
time: 2020/12/1 10:07
software: PyCharm

Description:
    stratified medium: 分层介质
    thickness: thickness of the layers in um
    epsilon: relative dielectric permittivity of the layers.
    polarisation: polarization state of field, 'TE' or 'TM'
    wavelength: the wavelength of the incident light in µm.
    kz: float, transverse wave vector in 1/µm.

    return:
        M: the transfer matrix of the medium.
"""

import numpy as np


class COPH:
    @staticmethod
    def transferMatrix(thickness, epsilon, polarization, wavelength, kz):
        # we know the shape of transfer function is (2, 2), and we know the epsilon may be either complex or float
        # so we could divide this into two kinds of situation, with complex and without complex
        # in numpy, np.int is equal to np.int32, np.float is equal to np.float64, np.complex is equal to np.complex128
        # but it is better to make clear declaration
        data_type = epsilon.dtype
        if data_type == "int32" or "float64":
            M = np.eye(2, dtype=np.float64)
        elif data_type == "complex128":
            M = np.eye(2, dtype=np.complex128)
        else:
            raise ValueError("The initialization of matrix M failed")

        # according to different mode of wave to get q value
        if polarization == 'TE' or 'Te' or 'te' or 'tE':
            q = np.ones_like(epsilon)
        elif polarization == 'TM' or 'Tm' or 'tm' or 'tM':
            q = 1.0 / epsilon
        else:
            raise ValueError("The input wave mode is not correct. It could be either 'TE' or 'TM'")

        # wave number in vacuum
        k0 = 2 * np.pi / wavelength
        # wave vector along x axis
        kx = np.sqrt(pow(k0, 2) * epsilon - pow(kz, 2))

        # iterate over layers
        for di, kxi, qi in zip(thickness, kx, q):
            cos_item = np.cos(kxi * di)
            sin_item = np.sin(kxi * di)
            m = np.array([[cos_item, 1 / qi / kxi * sin_item],
                          [-qi * kxi * sin_item, cos_item]])

            # M = np.dot(m, M)
            M = m @ M

        return M


if __name__ == '__main__':
    epsilon_ = np.array([2.25, 15.21, 2.25, 15.21, 2.25, 15.21, 2.25, 15.21, 2.25, 15.21])
    thickness_ = np.array([0.13, 0.05, 0.13, 0.05, 0.13, 0.05, 0.13, 0.05, 0.13, 0.05])
    polarization_ = 'TE'
    wavelength_ = 0.78
    kz_ = 0.0
    main = COPH()
    transfer_matrix = main.transferMatrix(thickness_, epsilon_, polarization_, wavelength_, kz_)
    print("The transfer_matrix is:")
    print(transfer_matrix)
"""
The transfer_matrix is:
[[-8.41653357e-03  7.55321107e-16]
 [-5.16929695e-13 -1.18813760e+02]]
"""
