import cupy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tqdm import tqdm


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


def iPSF_misalignment_k(xv, yv, k, x0, y0, zpp, E0, phi0, ma_theta, ma_phi):
    rpp = np.sqrt((xv - x0) ** 2 + (yv - y0) ** 2 + zpp ** 2)[..., None]  # from particle to the focal plane
    cos_theta = zpp / rpp  # cos of scattering angle

    phi_inc = k * zpp  # phase shift due to incedent OPD, (note that influence of zf is lumped into phi0)
    phi_sca = k * rpp  # phase shift due to return OPD
    fac = np.sqrt(1 + cos_theta ** 2) * 1 / (k * rpp)  # amplitude factor
    Escat = E0 * fac  # scattering amplitude

    ma_phi_cos = np.cos(ma_phi * np.pi / 180)
    ma_phi_sin = np.sin(ma_phi * np.pi / 180)
    ma_theta_sin = np.sin(ma_theta * np.pi / 180)
    ma = np.einsum('v,xy->xyv', k, (xv - x0) * ma_phi_cos + (yv - y0) * ma_phi_sin) * ma_theta_sin  # misalignment

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
            tif.write(dat[None, :, :, :, :].astype(np.float32).get())  # TZCYXS
        elif data.ndim == 3:
            d = np.moveaxis(data, [2, 1, 0], [0, 1, 2])
            tif.write(d[None, :, None, :, :].astype(np.float32).get())  # TZCYXS


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

    sigma = 0.1

    repeat = 10

    lam = 488

    a = 40
    mask_bkg = np.zeros_like(xv).astype(np.bool_)
    mask_bkg[a:-a, a:-a] = 1
    mask_bkg[90:110, 90:110] = 0

    b = 95
    mask_sig = np.zeros_like(xv).astype(np.bool_)
    mask_sig[b:-b, b:-b] = 1

    # Broadband
    bandwidth_list = np.arange(0, 100, 2.5)
    # bandwidth_list = np.arange(0, 200, 5)

    BB_noise_var_rcd = np.zeros((bandwidth_list.shape[0], repeat))

    for r in range(repeat):
        # generate noise map
        mean = 0
        std_dev = sigma
        noise = np.random.normal(mean, std_dev, xv.shape)
        noise = np.abs(noise)

        skip = 8

        for bandwidth_idx, bandwidth in tqdm(enumerate(bandwidth_list)):
            k_list = 2 * np.pi / (np.arange(lam - float(bandwidth) // 2, lam + float(bandwidth) // 2 + 1, 1) / 1.33)

            BB_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        BB_pattern_noise = BB_pattern_noise + noise[i, j] * iPSF_misalignment_k(xv, yv, k_list,
                                                                                                xv[i, j],
                                                                                                yv[i, j],
                                                                                                500, 0.3,
                                                                                                np.pi / 2, 0,
                                                                                                0)

            BB_noise_var_rcd[bandwidth_idx, r] = np.var(BB_pattern_noise[mask_bkg])

    # RO-iSCAT
    k = 2 * np.pi / (lam / 1.33)  # wavenumber in medium

    theta_list = np.arange(0, 40, 1)

    RO_noise_var_rcd = np.zeros((theta_list.shape[0], repeat))

    for r in range(repeat):
        # generate noise map
        mean = 0
        std_dev = sigma
        noise = np.random.normal(mean, std_dev, xv.shape)
        noise = np.abs(noise)

        skip = 8

        for theta_idx, theta in tqdm(enumerate(theta_list)):
            RO_pattern_signal = iPSF_misalignment_phi(xv, yv, k, 0, 0, 0, 0.3, np.pi / 2, theta, np.arange(0, 360, 10))
            RO_signal_var = np.var(RO_pattern_signal[mask_sig])

            RO_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        RO_pattern_noise = RO_pattern_noise + noise[i, j] * iPSF_misalignment_phi(xv, yv, k,
                                                                                                  xv[i, j],
                                                                                                  yv[i, j],
                                                                                                  500, 0.3,
                                                                                                  np.pi / 2, theta,
                                                                                                  np.arange(0, 360,
                                                                                                            10))

            RO_noise_var_rcd[theta_idx, r] = np.var(RO_pattern_noise[mask_bkg])

    plt.figure()
    plt.plot(bandwidth_list.get(), BB_noise_var_rcd.get().mean(axis=-1), 'r')
    plt.plot(theta_list.get(), RO_noise_var_rcd.get().mean(axis=-1), 'r')
    plt.show()
