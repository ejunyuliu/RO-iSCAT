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
            tif.save(dat[None, :, :, :, :].astype(np.float32))  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.save(d[None, :, None, :, :].astype(np.float32))  # TZCYXS


if __name__ == '__main__':
    # Set to number of cores on your machine
    CORES = 4
    plt.rcParams['figure.figsize'] = [9.5, 6]

    # DEFINE CONSTANTS
    k = 2 * np.pi / (635 / 1.33)  # wavenumber in medium
    fov = 4000  # size of field of view

    # DEFINE MESH
    nx = 200
    ny = 200
    xv, yv = getMesh(-0.5, 0.5, -0.5, 0.5, nx=nx, ny=ny)
    xv = xv * fov
    yv = yv * fov

    sigma_list = np.arange(0, 0.2, 0.005)
    repeat_num = 1

    RO_noise_var_list = np.zeros((len(sigma_list), repeat_num))
    RO_SNR_list = np.zeros((len(sigma_list), repeat_num))
    noRO_noise_var_list = np.zeros((len(sigma_list), repeat_num))
    noRO_SNR_list = np.zeros((len(sigma_list), repeat_num))

    a = 40
    mask_bkg = np.zeros_like(xv).astype(np.bool_)
    mask_bkg[a:-a, a:-a] = 1
    mask_bkg[90:110, 90:110] = 0

    b = 95
    mask_sig = np.zeros_like(xv).astype(np.bool_)
    mask_sig[b:-b, b:-b] = 1

    theta = 22
    print(theta)

    noRO_pattern_signal = iPSF_misalignment(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, 0, np.array([0]))
    RO_pattern_signal = iPSF_misalignment(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, theta, np.arange(0, 360, 10))

    noRO_signal_var = np.var(noRO_pattern_signal[mask_sig])
    RO_signal_var = np.var(RO_pattern_signal[mask_sig])

    for sigma_idx, sigma in tqdm(enumerate(sigma_list)):
        for repeat_idx in range(repeat_num):
            # generate noise map
            mean = 0
            std_dev = sigma
            noise = np.random.normal(mean, std_dev, xv.shape)
            noise = np.abs(noise)

            skip = 8

            noRO_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        noRO_pattern_noise = noRO_pattern_noise + noise[i, j] * iPSF_misalignment(xv, yv, k, xv[i, j],
                                                                                                  yv[i, j],
                                                                                                  1000, 0.3,
                                                                                                  np.pi / 2, 0,
                                                                                                  np.array([0]))

            noRO_pattern = noRO_pattern_signal + noRO_pattern_noise
            noRO_noise_var = np.var(noRO_pattern[mask_bkg])

            noRO_noise_var_list[sigma_idx, repeat_idx] = noRO_noise_var
            noRO_SNR_list[sigma_idx, repeat_idx] = noRO_signal_var / noRO_noise_var

            RO_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        RO_pattern_noise = RO_pattern_noise + noise[i, j] * iPSF_misalignment(xv, yv, k, xv[i, j],
                                                                                              yv[i, j],
                                                                                              1000, 0.3,
                                                                                              np.pi / 2, theta,
                                                                                              np.arange(0, 360, 10))
            RO_pattern = RO_pattern_signal + RO_pattern_noise
            RO_noise_var = np.var(RO_pattern[mask_bkg])

            RO_noise_var_list[sigma_idx, repeat_idx] = RO_noise_var
            RO_SNR_list[sigma_idx, repeat_idx] = RO_signal_var / RO_noise_var

    plt.figure()
    plt.plot(sigma_list.get(), noRO_noise_var_list[:, 0].get(), 'r')
    plt.plot(sigma_list.get(), RO_noise_var_list[:, 0].get(), 'b')
    plt.show()

    plt.figure()
    plt.plot(sigma_list.get(), noRO_SNR_list[:, 0].get(), 'r')
    plt.plot(sigma_list.get(), RO_SNR_list[:, 0].get(), 'b')
    plt.show()
