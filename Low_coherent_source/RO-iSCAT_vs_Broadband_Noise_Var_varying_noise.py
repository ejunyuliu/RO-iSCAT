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

    sigma_list = np.arange(0, 0.21, 0.01)

    phi0 = 0
    k0 = 2 * np.pi / (488 / 1.33)  # wavenumber in medium

    phi_list = np.arange(0, 360, 10)
    k_list = 2 * np.pi / (np.arange(450, 500, 5) / 1.33)

    BB_pattern_signal = iPSF_misalignment_k(xv, yv, k_list, 0, 0, 0, 0.3, np.pi / 2, 0, phi0)
    RO_pattern_signal = iPSF_misalignment_phi(xv, yv, k0, 0, 0, 0, 0.3, np.pi / 2, 22, phi_list)
    iSCAT_pattern_signal = iPSF_misalignment_phi(xv, yv, k0, 0, 0, 0, 0.3, np.pi / 2, 22, np.array([0]))

    repeat = 10
    BB_var_rcd = np.zeros((sigma_list.shape[0], repeat))
    RO_var_rcd = np.zeros((sigma_list.shape[0], repeat))
    iSCAT_var_rcd = np.zeros((sigma_list.shape[0], repeat))

    for sigma_idx, sigma in tqdm(enumerate(sigma_list)):
        for r in range(repeat):
            # generate noise map
            mean = 0
            std_dev = sigma
            noise = np.random.normal(mean, std_dev, xv.shape)
            noise = np.abs(noise)

            skip = 8

            iSCAT_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        iSCAT_pattern_noise = iSCAT_pattern_noise + noise[i, j] * iPSF_misalignment_phi(xv, yv, k0,
                                                                                                        xv[i, j],
                                                                                                        yv[i, j],
                                                                                                        500, 0.3,
                                                                                                        np.pi / 2, 22,
                                                                                                        np.array([0]))

            iSCAT_var_rcd[sigma_idx, r] = np.var(iSCAT_pattern_noise)
            iSCAT_pattern = iSCAT_pattern_signal + iSCAT_pattern_noise

            BB_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        BB_pattern_noise = BB_pattern_noise + noise[i, j] * iPSF_misalignment_k(xv, yv, k_list,
                                                                                                xv[i, j],
                                                                                                yv[i, j],
                                                                                                500, 0.3,
                                                                                                np.pi / 2, 0,
                                                                                                phi0)

            BB_var_rcd[sigma_idx, r] = np.var(BB_pattern_noise)
            BB_pattern = BB_pattern_signal + BB_pattern_noise

            RO_pattern_noise = 0
            for i in range(nx):
                for j in range(ny):
                    if i % skip == 0 & j % skip == 0:
                        RO_pattern_noise = RO_pattern_noise + noise[i, j] * iPSF_misalignment_phi(xv, yv, k0, xv[i, j],
                                                                                                  yv[i, j],
                                                                                                  500, 0.3,
                                                                                                  np.pi / 2, 22,
                                                                                                  phi_list)

            RO_var_rcd[sigma_idx, r] = np.var(RO_pattern_noise)
            RO_pattern = RO_pattern_signal + RO_pattern_noise

            noise_mask = np.zeros_like(noise)
            noise_mask[::skip, ::skip] = 1

            if sigma in [0, 0.05, 0.1, 0.15, 0.2]:
                if r == 0:
                    save_tiff('./pattern/' + str(sigma) + '_noise.tif', (noise_mask * noise)[..., None])
                    save_tiff('./pattern/' + str(sigma) + '_iSCAT_pattern.tif', iSCAT_pattern[..., None])
                    save_tiff('./pattern/' + str(sigma) + '_RO_pattern.tif', RO_pattern[..., None])
                    save_tiff('./pattern/' + str(sigma) + '_BB_pattern.tif', BB_pattern[..., None])

    plt.figure()
    plt.plot(sigma_list.get(), iSCAT_var_rcd.get(), 'g')
    plt.plot(sigma_list.get(), BB_var_rcd.get(), 'r')
    plt.plot(sigma_list.get(), RO_var_rcd.get(), 'b')
    plt.show()
