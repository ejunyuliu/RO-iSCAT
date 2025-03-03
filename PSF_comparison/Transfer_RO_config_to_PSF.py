import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy.special import j1
from scipy.ndimage import shift
from scipy.signal import convolve2d
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def read_tiff(filename):
    with tifffile.TiffFile(filename) as tf:
        tif = np.ascontiguousarray(np.moveaxis(tf.asarray(), [0, 1, 2], [2, 1, 0]))

    return tif


def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


def Gaussian_PSF(X, Y, Z, fwhm_x, fwhm_y, fwhm_z):
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    sigma_z = fwhm_z / (2 * np.sqrt(2 * np.log(2)))

    psf = np.exp(-((X ** 2) / (2 * sigma_x ** 2) +
                   (Y ** 2) / (2 * sigma_y ** 2) +
                   (Z ** 2) / (2 * sigma_z ** 2)))

    return psf


def calculate_fwhm(x, y):
    peak_idx = np.argmax(y)
    peak_y = y[peak_idx]
    half_max = peak_y / 2

    left_idx = np.where(y[:peak_idx] <= half_max)[0][-1]
    right_idx = np.where(y[peak_idx:] <= half_max)[0][0] + peak_idx

    # 计算 FWHM
    x1, x2 = x[left_idx], x[right_idx]
    fwhm = x2 - x1

    return fwhm


if __name__ == '__main__':
    NA = 0.8
    lam = 488
    n = 1.33

    # generate grid
    pixel_size = 10
    size = 512

    axis_seq = np.arange(-size // 2, size - size // 2) * pixel_size

    X, Y, Z = np.meshgrid(axis_seq, axis_seq, axis_seq, indexing='ij')

    # generate confocal PSF
    # fwhm_lateral = 0.51 * lam / NA
    # fwhm_axial = 1.4 * lam * n / (NA ** 2)

    # Ref: The Intermediate Optical System of Laser-Scanning Confocal Microscopes
    if NA == 0.8:
        fwhm_lateral = 250
        fwhm_axial = 1320

    if NA == 1.4:
        fwhm_lateral = 140
        fwhm_axial = 380

    psf_confocal = Gaussian_PSF(X, Y, Z, fwhm_lateral, fwhm_lateral, fwhm_axial)

    # generate widefield PSF
    # fwhm_lateral = 0.51 * lam / NA
    # fwhm_axial = 1.4 * lam * n / (NA ** 2) * np.sqrt(2)

    if NA == 0.8:
        fwhm_lateral = 350
        fwhm_axial = 1870

    if NA == 1.4:
        fwhm_lateral = 190
        fwhm_axial = 540

    psf_widefield = Gaussian_PSF(X, Y, Z, fwhm_lateral, fwhm_lateral, fwhm_axial)

    save_tiff('./PSF/widefield.tif', psf_widefield)
    save_tiff('./PSF/confocal.tif', psf_confocal)

    # generate RO-iSCAT PSF
    # rat = 0.40283650647
    # psf_ro = np.zeros_like(psf_wf)
    # for z in range(0, size):
    #     for phi in range(0, 360, 10):
    #         shift_x = np.abs(z - size // 2) * rat * pixel_size * np.sin(phi / 180 * np.pi)
    #         shift_y = np.abs(z - size // 2) * rat * pixel_size * np.cos(phi / 180 * np.pi)
    #         psf_ro[..., z] = psf_ro[..., z] + shift(psf_wf[..., z], shift=(shift_y, shift_x))

    # generate RO PSF
    rat = 0.40283650647
    mask = np.zeros_like(psf_widefield)
    for z in range(0, size):
        R = np.sqrt(X[..., z] ** 2 + Y[..., z] ** 2)
        Z = np.abs(z - size // 2)
        mask[..., z][(R <= rat * (Z + 3) * pixel_size) & (R >= rat * Z * pixel_size)] = 1
        mask[..., z] = mask[..., z] / (mask[..., z].sum() + 1e-7)

    save_tiff('./PSF/mask.tif', mask)

    otf_mask = np.fft.fftn(mask, axes=(0, 1))
    otf_wf = np.fft.fftn(psf_widefield, axes=(0, 1))
    otf_ro = otf_mask * otf_wf
    psf_ro = np.real(np.fft.fftshift(np.fft.ifftn(otf_ro, axes=(0, 1)), axes=(0, 1)))

    save_tiff('./PSF/rotational-oblique.tif', psf_ro)

    # check PSF profiles

    psf_widefield = psf_widefield / psf_widefield.max()
    psf_confocal = psf_confocal / psf_confocal.max()
    psf_ro = psf_ro / psf_ro.max()

    wf_axial = psf_widefield[size // 2, size // 2, :]
    cf_axial = psf_confocal[size // 2, size // 2, :]
    ro_axial = psf_ro[size // 2, size // 2, :]

    print('axial')
    print(calculate_fwhm(axis_seq, wf_axial))
    print(calculate_fwhm(axis_seq, cf_axial))
    print(calculate_fwhm(axis_seq, ro_axial))

    plt.figure()
    plt.plot(axis_seq, wf_axial, 'g')
    plt.plot(axis_seq, cf_axial, 'k')
    plt.plot(axis_seq, ro_axial, 'r')
    plt.show()

    wf_lateral = psf_widefield[size // 2, :, size // 2]
    cf_lateral = psf_confocal[size // 2, :, size // 2]
    ro_lateral = psf_ro[size // 2, :, size // 2]

    print('lateral')
    print(calculate_fwhm(axis_seq, wf_lateral))
    print(calculate_fwhm(axis_seq, cf_lateral))
    print(calculate_fwhm(axis_seq, ro_lateral))

    plt.figure()
    plt.plot(axis_seq, wf_lateral, 'g')
    plt.plot(axis_seq, cf_lateral, 'k')
    plt.plot(axis_seq, ro_lateral, 'r')
    plt.show()
