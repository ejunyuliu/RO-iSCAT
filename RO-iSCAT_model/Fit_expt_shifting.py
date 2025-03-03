# DEFINE iPSF

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import h5py
from pathlib import Path


# DEFINE MESH
def getMesh(xmin, xmax, ymin, ymax, nx=2000, ny=2000):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma = (k * ((xv - x0) * np.cos(ma_phi * np.pi / 180) + (yv - y0) * np.sin(ma_phi * np.pi / 180)) *
          np.sin(ma_theta * np.pi / 180))  # misalignment

    phi_diff = ma - (phi0 + phi_inc + phi_sca)  # phase difference
    # phi_diff = ma
    # phi_diff = - (phi0 + phi_inc)
    # phi_diff = - phi_sca
    # phi_diff = ma - phi_sca

    iPSF = phi_diff
    # iPSF = Escat
    # iPSF = 2 * Escat * np.cos(phi_diff)
    # iPSF = 2 * np.cos(phi_diff)

    return iPSF


if __name__ == '__main__':
    # DEFINE CONSTANTS
    k = 2 * np.pi / (488 / 1.33)  # wavenumber in medium
    fov = 14600  # size of field of view

    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
    xv = xv * fov
    yv = yv * fov

    plt.clf()

    dz_list = [0,
               294.1176471,
               588.2352941,
               882.3529412,
               1176.470588,
               1470.588235,
               1764.705882,
               2058.823529,
               2352.941176,
               2647.058824,
               2941.176471,
               3235.294118,
               3529.411765,
               3823.529412,
               4117.647059,
               4411.764706,
               4705.882353,
               5000,
               5294.117647,
               5588.235294,
               5882.352941,
               6176.470588,
               6470.588235,
               6764.705882,
               7058.823529,
               7352.941176,
               7647.058824,
               7941.176471,
               8235.294118,
               8529.411765,
               8823.529412,
               9117.647059,
               9411.764706,
               ]

    r_list = []

    for dz in dz_list:
        iPSF = iPSF_misalignment(xv, yv, k, 0, 0, dz, 0.3, np.pi / 2, 22, 0)

        max_position = np.unravel_index(np.argmax(iPSF), iPSF.shape)
        xb, yb = xv[max_position[0], max_position[1]], yv[max_position[0], max_position[1]]
        r_list.append(np.sqrt(xb ** 2 + yb ** 2))

    plt.figure()
    plt.plot(dz_list, r_list)
    plt.xlabel('$x$ [nm]')
    plt.ylabel('$y$ [nm]')
    plt.show()
    plt.clf()
