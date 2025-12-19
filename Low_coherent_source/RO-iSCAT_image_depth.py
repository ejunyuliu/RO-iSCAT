import cupy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tqdm import tqdm


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


def iPSF_misalignment_phi(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)[..., None]  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle
    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma_phi_cos = np.cos(ma_phi * np.pi / 180)
    ma_phi_sin = np.sin(ma_phi * np.pi / 180)
    ma_theta_sin = np.sin(ma_theta * np.pi / 180)
    ma = k * (np.einsum('xy,v->xyv', xv - x0, ma_phi_cos) + np.einsum('xy,v->xyv', yv - y0,
                                                                      ma_phi_sin)) * ma_theta_sin  # misalignment

    phi_diff = ma - (phi0 + phi_inc + phi_sca)  # phase difference

    iPSF = 2 * Escat * np.cos(phi_diff)

    return iPSF.mean(axis=-1)


def getMesh(xmin, xmax, ymin, ymax, nx=200, ny=200):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return np.meshgrid(x, y)


if __name__ == '__main__':
    # Set to number of cores on your machine
    CORES = 4
    plt.rcParams['figure.figsize'] = [9.5, 6]

    # DEFINE CONSTANTS
    fov = 4000  # size of field of view

    # DEFINE MESH
    nx = 200
    ny = 200
    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5, nx=nx, ny=ny)
    xv = xv * fov
    yv = yv * fov

    lam = 488

    # RO-iSCAT
    k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium

    theta_list = np.arange(0, 40, 1)
    zpp_list = np.arange(0, 5000, 100)

    rcd = np.zeros((len(theta_list), len(zpp_list)))

    for theta_idx, theta in tqdm(enumerate(theta_list)):
        for zpp_idx, zpp in enumerate(zpp_list):
            iPSF = iPSF_misalignment_phi(xv, yv, k, 0, 0, zpp, 0.3, np.pi / 2, theta, np.arange(0, 360, 10))

            rcd[theta_idx, zpp_idx] = np.var(iPSF)

    save_tiff('depth.tif', rcd)
