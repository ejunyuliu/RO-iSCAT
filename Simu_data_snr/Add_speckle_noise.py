import cupy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tqdm import tqdm


def iPSF_misalignment(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
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


def save_tiff(filename, data):
    with tifffile.TiffWriter(filename, imagej=True) as tif:
        if data.ndim == 4:
            dat = np.moveaxis(data, [2, 3, 1, 0], [0, 1, 2, 3])
            tif.save(dat[None, :, :, :, :].astype(np.float32).get())  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32).get())  # TZCYXS


if __name__ == '__main__':
    # DEFINE CONSTANTS
    k = 2 * np.pi / (635 / 1.33)  # wavenumber in medium
    fov = 4000  # size of field of view

    # DEFINE MESH
    nx = 200
    ny = 200
    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5, nx=nx, ny=ny)
    xv = xv * fov
    yv = yv * fov

    a = 40
    mask_bkg = np.zeros_like(xv).astype(np.bool_)
    mask_bkg[a:-a, a:-a] = 1
    mask_bkg[90:110, 90:110] = 0

    b = 95
    mask_sig = np.zeros_like(xv).astype(np.bool_)
    mask_sig[b:-b, b:-b] = 1

    # generate noise map
    mean = 0
    std_dev = 0.05
    noise = np.random.normal(mean, std_dev, xv.shape)
    noise = np.abs(noise)

    skip = 8

    noise_mask = np.zeros_like(noise)
    noise_mask[::skip, ::skip] = 1
    save_tiff(filename='noise_mask.tif', data=(noise_mask * noise)[...,None])

    noRO_pattern_signal = iPSF_misalignment(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, 0, np.array([0]))
    save_tiff(filename='noRO_GT.tif', data=noRO_pattern_signal[..., None])
    noRO_pattern_noise = 0
    for i in tqdm(range(nx)):
        for j in range(ny):
            if i % skip == 0 & j % skip == 0:
                noRO_pattern_noise = noRO_pattern_noise + noise[i, j] * iPSF_misalignment(xv, yv, k, xv[i, j], yv[i, j],
                                                                                          1000, 0.3,
                                                                                          np.pi / 2, 0, np.array([0]))
    noRO_pattern = noRO_pattern_signal + noRO_pattern_noise
    save_tiff(filename='noRO_pattern.tif', data=noRO_pattern[...,None])
    print(np.var(noRO_pattern_signal[mask_sig])/np.var(noRO_pattern[mask_bkg]))
    plt.imshow(noRO_pattern.get())
    plt.show()

    RO_pattern_signal = iPSF_misalignment(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, 22, np.arange(0, 360, 10))
    save_tiff(filename='RO_GT.tif', data=RO_pattern_signal[..., None])
    RO_pattern_noise = 0
    for i in tqdm(range(nx)):
        for j in range(ny):
            if i % skip == 0 & j % skip == 0:
                RO_pattern_noise = RO_pattern_noise + noise[i, j] * iPSF_misalignment(xv, yv, k, xv[i, j], yv[i, j],
                                                                                      1000, 0.3,
                                                                                      np.pi / 2, 22,
                                                                                      np.arange(0, 360, 10))

    RO_pattern = RO_pattern_signal + RO_pattern_noise
    save_tiff(filename='RO_pattern.tif', data=RO_pattern[...,None])
    print(np.var(RO_pattern_signal[mask_sig])/np.var(RO_pattern[mask_bkg]))
    plt.imshow(RO_pattern.get())
    plt.show()
