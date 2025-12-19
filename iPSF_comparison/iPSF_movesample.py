import numpy as np
import matplotlib.pyplot as plt
import tifffile
import h5py
from pathlib import Path
import bisect


def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


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

    phi_diff = ma - (phi_inc + phi_sca)  # phase difference
    iPSF = 2 * Escat * np.cos(phi_diff)

    return iPSF


if __name__ == '__main__':
    # DEFINE CONSTANTS
    lam = 488
    n = 1.33
    k = 2 * np.pi / (lam / n)  # wavenumber in medium

    Xsize = 200
    Ysize = 200
    Zsize = 800
    vox_dim = 10

    fov = vox_dim * Xsize

    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5, nx=Xsize, ny=Ysize)
    xv = xv * fov
    yv = yv * fov

    zpp_list = (np.arange(0, Zsize) - Zsize // 2) * vox_dim
    # zpp_list = np.arange(0, Zsize) * vox_dim

    x_list = (np.arange(0, Xsize) - Xsize // 2) * vox_dim

    # Generate
    iPSF_noRO = np.zeros((Xsize, Ysize, Zsize))
    iPSF_noRO_FWHM = np.zeros((Zsize))

    for zpp_idx, zpp in enumerate(zpp_list):
        iPSF = iPSF_misalignment(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, 0, 0)
        iPSF_noRO[..., zpp_idx] = iPSF

    # Generate
    phi_list = range(0, 360, 10)
    iPSF_RO = np.zeros((Xsize, Ysize, Zsize))
    iPSF_RO_FWHM = np.zeros((Zsize))

    for zpp_idx, zpp in enumerate(zpp_list):
        iPSF = 0
        for phi_idx, phi in enumerate(phi_list):
            iPSF = iPSF + iPSF_misalignment(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, 22, phi)
        iPSF_RO[..., zpp_idx] = iPSF


    save_tiff('wf.tif', np.real(iPSF_noRO))
    save_tiff('ro.tif', np.real(iPSF_RO))
