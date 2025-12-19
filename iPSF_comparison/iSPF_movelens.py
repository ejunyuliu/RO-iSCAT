import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path


# DEFINE iPSF
def iPSF_no_beam(xv, yv, k, x0, y0, zpp, E0, phi0):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    if zpp >= 0:
        phi_diff = -(phi0 + phi_inc + phi_sca)  # phase difference
        iPSF = 2 * Escat * np.cos(phi_diff)
    else:
        phi_diff = phi0 + phi_inc + phi_sca
        iPSF = 2 * Escat * np.cos(phi_diff)

    return iPSF


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma = k * ((xv - x0) * np.cos(ma_phi * np.pi / 180) + (yv - y0) * np.sin(ma_phi * np.pi / 180)) * np.sin(
        ma_theta * np.pi / 180)  # misalignment

    if zpp >= 0:
        phi_diff = ma - (phi0 + phi_inc + phi_sca)  # phase difference
    else:
        phi_diff = - ma - phi0 - phi_inc + phi_sca

    iPSF = 2 * Escat * np.cos(phi_diff)

    return iPSF


def RO_iPSF(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta):
    iPSF = 0
    for phi in range(0, 360, 10):
        iPSF = iPSF + iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, phi)

    return iPSF


def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


if __name__ == '__main__':
    # DEFINE CONSTANTS
    k = 2 * np.pi / (635 / 1.33)  # wavenumber in medium
    fov = 4000  # size of field of view

    # DEFINE MESH
    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5)
    xv = xv * fov
    yv = yv * fov

    zp = 3000  # axial position of particle
    zfs = np.linspace(zp // 2, zp // 2 * 3, 500)  # axial positions of the focal plane
    zpps = zp - zfs  # axial positions of the particle w.r.t. the focal plane
    phi0s = np.pi * 1.5 + 2 * k * zfs / np.cos(20 / 180 * np.pi)  # phi0 given the location of zf

    iPSF_wf = np.array(
        [iPSF_misalignment(xv, yv, k, 0, 0, zpps[i], 0.3, phi0s[i], 20, 0) for i in range(len(zfs))])

    iPSF_ro = np.array(
        [RO_iPSF(xv, yv, k, 0, 0, zpps[i], 0.3, phi0s[i], 22) for i in range(len(zfs))])

    save_tiff('wf.tif', iPSF_wf)
    save_tiff('ro.tif', iPSF_ro)
