# -*- coding:utf-8 -*-
"""
author: 15025
age: 26
e-mail: 1502506285@qq.com
time: 2020/12/1 10:07
software: PyCharm

Description:
    thickness: thickness of the layers in um
    epsilon: relative dielectric permittivity of the layers.
    polarisation: str, polarization state of field, 'TE' or 'TM'
    wavelength: the wavelength of the incident light in µm.
    angle_incident: the angle of incidence in degree(not radian)
    n_in, n_out : float, The refractive indices of the input and output layers.

    return:
        t: 1D array, transmitted amplitude
        r: 1D array, reflected amplitude
        T: 1D array, transmitted energy
        R: 1D array, reflected energy
"""

import numpy as np
import transferMatrix
import matplotlib.pyplot as plt


class COPH:
    @staticmethod
    def spectrum(thickness, epsilon, polarization, wavelength, angle_inc, n_in, n_out):
        # epsilon of input and output layers
        epsilon_in = pow(n_in, 2)
        epsilon_out = pow(n_out, 2)

        # according to different mode of wave to get q value
        if polarization == 'TE' or 'Te' or 'te' or 'tE':
            q_in = 1
            q_out = 1
        elif polarization == 'TM' or 'Tm' or 'tm' or 'tM':
            q_in = 1.0 / epsilon_in
            q_out = 1.0 / epsilon_out
        else:
            raise ValueError("The input wave mode is not correct. It could be either 'TE' or 'TM'")

        # wave number in vacuum, wave vector has direction
        k0 = 2 * np.pi / wavelength
        # wave vector along z axis
        kz = k0 * np.sqrt(n_in) * np.sin(np.deg2rad(angle_inc))
        # wave vector along x axis
        kx_in = np.sqrt(pow(k0, 2) * epsilon_in - pow(kz, 2))
        kx_out = np.sqrt(pow(k0, 2) * epsilon_out - pow(kz, 2))

        # we need to get all value of r and t, so we need to initialize an array to store them.
        # we may study different wavelength situation
        denominator = np.zeros(wavelength.shape, dtype=np.complex128)
        numerator = np.zeros(wavelength.shape, dtype=np.complex128)

        # iterate over layers
        # we will introduce i here, so the final result will definitely be complex
        # attention, kz also need to be iterated
        for index, (lambda_, kx_ini, kx_outi, kz_i) in enumerate(zip(wavelength, kx_in, kx_out, kz)):
            # the transfer matrix depend on wavelength and thickness, so it is changeable
            M = transferMatrix.COPH.transferMatrix(thickness, epsilon, polarization, lambda_, kz_i)
            denominator[index] = q_in * kx_ini * M[1, 1] + q_out * kx_outi * M[0, 0] + 1j * (
                    M[1, 0] - q_in * kx_ini * q_out * kx_outi * M[0, 1])

            numerator[index] = q_in * kx_ini * M[1, 1] - q_out * kx_outi * M[0, 0] - 1j * (
                    M[1, 0] + q_in * kx_ini * q_out * kx_outi * M[0, 1])

        r = numerator / denominator
        t = 2 * q_in * kx_in / denominator
        R = np.real(r * np.conj(r))
        T = q_out * np.real(kx_out) / (q_in * np.real(kx_in)) * np.real(t * np.conj(t))

        return r, t, R, T


if __name__ == '__main__':
    # set testing parameters
    epsilon_ = np.array([2.25, 15.21, 2.25, 15.21, 2.25, 15.21, 2.25, 15.21, 2.25, 15.21])
    thickness_ = np.array([0.13, 0.05, 0.13, 0.05, 0.13, 0.05, 0.13, 0.05, 0.13, 0.05])
    n_in_ = 1
    n_out_ = 1.5
    angle_inc_ = 0
    polarization_ = 'TE'
    wavelength_ = np.linspace(0.5, 1.5, 1001)
    main = COPH()
    r_, t_, R_, T_ = main.spectrum(thickness_, epsilon_, polarization_, wavelength_, angle_inc_, n_in_, n_out_)
    # plot figure
    plt.figure()
    plt.plot(wavelength_, T_, wavelength_, R_)
    plt.xlabel('wavelength [um]')
    plt.ylabel('reflectance, transmittance')
    plt.legend(['transmittance', 'reflectance'], loc='center', frameon=False)
    plt.show()
