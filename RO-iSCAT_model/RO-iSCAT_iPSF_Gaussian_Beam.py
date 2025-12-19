# DEFINE iPSF

import cupy as np
import matplotlib.pyplot as plt
import tifffile
import pymc as pm
import arviz as az
import xarray as xr
import seaborn as sns
import h5py
from pathlib import Path
import tifffile


def save_tiff(filename='sh.tif', data=None):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.write(dat[None, :, :, :, :].get().astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.write(d[None, :, None, :, :].get().astype(np.float32))  # TZCYXS
        elif data.ndim == 2:
            data = data[:, :, None]
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.write(d[None, :, None, :, :].get().astype(np.float32))  # TZCYXS


# DEFINE MESH
def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    ma_theta = ma_theta / 180 * np.pi
    ma_phi = ma_phi / 180 * np.pi

    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma = k * ((xv - x0) * np.cos(ma_phi) + (yv - y0) * np.sin(ma_phi)) * np.sin(ma_theta)  # misalignment

    phi_diff = ma - (phi_inc + phi_sca)  # phase difference

    iPSF = 2 * Escat * np.cos(phi_diff)

    return iPSF


def iPSF_misalignment_Gaussian(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi, divergence=1e-1):
    ma_theta = ma_theta / 180 * np.pi
    ma_phi = ma_phi / 180 * np.pi

    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma_theta_list = np.linspace(ma_theta, ma_theta + 3 * divergence, num=1000)
    ma_phi_list = np.linspace(ma_phi, ma_phi, num=1)

    ma_theta_mesh, ma_phi_mesh = np.meshgrid(ma_theta_list, ma_phi_list)

    A = np.exp(
        - ((ma_theta_mesh - ma_theta) ** 2 + (np.sin(ma_theta) ** 2) * (ma_phi_mesh - ma_phi) ** 2) / (divergence ** 2)
    )

    ma_tmp = np.einsum('xy,tp->xytp', xv - x0, np.cos(ma_phi_mesh)) + \
             np.einsum('xy,tp->xytp', yv - y0, np.sin(ma_phi_mesh))
    ma = k * np.einsum('xytp,tp->xytp', ma_tmp, np.sin(ma_theta_mesh))  # misalignment

    phi_diff = ma - (phi_inc + phi_sca)[:, :, None, None]  # phase difference

    iPSF = 2 * np.einsum('xy,xytp,tp->xytp', Escat, np.cos(phi_diff), A)

    return iPSF.mean(axis=(2, 3))


if __name__ == '__main__':
    # DEFINE CONSTANTS
    k = 2 * np.pi / (488 / 1.33)  # wavenumber in medium
    fov = 14600  # size of field of view

    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
    xv = xv * fov
    yv = yv * fov

    # Generate
    iPSF = np.zeros_like(xv)

    theta = 22

    for dvg in [1e-3, 1e-2, 1e-1, 1]:
        for zpp in [100, 1000, 10000]:
            iPSF_RO_Gaussian = 0
            iPSF_RO_nonGaussian = 0
            for phi in range(0, 360, 2):
                iPSF_Gaussian = iPSF_misalignment_Gaussian(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, theta, phi,
                                                           divergence=dvg)
                iPSF_RO_Gaussian = iPSF_RO_Gaussian + iPSF_Gaussian

                iPSF_nonGaussian = iPSF_misalignment(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, theta, phi)
                iPSF_RO_nonGaussian = iPSF_RO_nonGaussian + iPSF_nonGaussian

            save_tiff('dvg=' + str(dvg) + ', zpp=' + str(zpp) + ' Gaussian.tif', iPSF_RO_Gaussian)
            save_tiff('dvg=' + str(dvg) + ', zpp=' + str(zpp) + ' nonGaussian.tif', iPSF_RO_nonGaussian)
