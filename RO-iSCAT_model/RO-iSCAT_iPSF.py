# DEFINE iPSF

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import h5py
from pathlib import Path


# DEFINE MESH
def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
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

    ma = k * ((xv - x0) * np.cos(ma_phi * np.pi / 180) + (yv - y0) * np.sin(ma_phi * np.pi / 180)) * np.sin(
        ma_theta * np.pi / 180)  # misalignment

    phi_ma = ma
    phi_de = - (phi_inc + phi_sca)
    phi_diff = ma - (phi_inc + phi_sca)  # phase difference

    iPSF_cos = np.cos(phi_diff)
    iPSF = 2 * Escat * np.cos(phi_diff)

    return phi_ma, phi_de, phi_diff, iPSF_cos, iPSF


if __name__ == '__main__':
    # DEFINE CONSTANTS
    k = 2 * np.pi / (488 / 1.33)  # wavenumber in medium
    fov = 14600  # size of field of view

    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
    xv = xv * fov
    yv = yv * fov

    # Generate
    iPSF = np.zeros_like(xv)

    iPSF_RO = 0
    zpp = 10
    for phi in range(0, 360, 5):
        phi_ma, phi_de, phi_diff, iPSF_cos, iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, 22, phi)
        iPSF_RO = iPSF_RO + iPSF

    plt.figure()
    plt.imshow(iPSF)
    plt.title('iSCAT iPSF')
    plt.show()

    plt.figure()
    plt.imshow(iPSF_RO)
    plt.title('RO-iSCAT iPSF')
    plt.show()

