import numpy as np
import scipy.constants as ct
from wake_t.physics_models.laser.laser_pulse import GaussianPulse


def calculate_spot_size(a_env, dr):
    # Project envelope to r
    a_proj = np.sum(np.abs(a_env), axis=0)

    # Maximum is on axis
    a_max = a_proj[0]

    # Get first index of value below a_max / e
    i_first = np.where(a_proj <= a_max / np.e)[0][0]

    # Do linear interpolation to get more accurate value of w.
    # We build a line y = a + b*x, where:
    #     b = (y_2 - y_1) / (x_2 - x_1)
    #     a = y_1 - b*x_1
    #
    #     y_1 is the value of a_proj at i_first - 1
    #     y_2 is the value of a_proj at i_first
    #     x_1 and x_2 are the radial positions of y_1 and y_2
    #
    # We can then determine the spot size by interpolating between y_1 and y_2,
    # that is, do x = (y - a) / b, where y = a_max/e
    y_1 = a_proj[i_first - 1]
    y_2 = a_proj[i_first]
    x_1 = (i_first - 1) * dr + dr / 2
    x_2 = i_first * dr + dr / 2
    b = (y_2 - y_1) / (x_2 - x_1)
    a = y_1 - b * x_1
    w = (a_max / np.e - a) / b
    return w


def susceptibility(np, r, r0, dn):
    return 1 + (dn * r ** 2 / r0 ** 2) / np


def spotsize(dn_c, dn, r_0, w_0, k_os, z):
    # returns w
    return np.sqrt((1 + dn_c * r_0 ** 4 / (dn * w_0 ** 4) + (
            1 - dn_c * r_0 ** 4 / (dn * w_0 ** 4)) * np.cos(
        k_os * z)) * w_0 ** 2 / 2)

def test_max_error():
    # Plasma density
    n_p = 1e23  # m^{-3}

    # Laser parameters in SI units
    tau = 25e-15  # s
    w_0 = 40e-6  # m
    l_0 = 0.8e-6  # m
    z_c = 0.  # m
    a_0 = 3

    # Create laser pulse
    laser = GaussianPulse(z_c, l_0=l_0, w_0=w_0, a_0=a_0, tau=tau,
                          z_foc=0, polarization='circular')

    # Grid and time parameters
    zmin = -100e-6  # m
    zmax = 100e-6  # m
    rmax = 200e-6  # m
    z_per_pl = 40  # grid points per pulse length
    nz = int((zmax - zmin) * z_per_pl / (tau * ct.c))
    r_per_pw = 55  # grid points per pulse width
    nr = int(rmax * r_per_pw / w_0)
    dr = rmax / nr
    t_max = 10e-2 / ct.c  # s
    nt = 1000
    dt = t_max / nt

    # Plasma susceptibility
    r = np.linspace(dr / 2, rmax - dr / 2, nr)
    dn_c = 1 / (np.pi * ct.value(
        "classical electron radius") * w_0 ** 2)
    dn = 2 * dn_c
    chi_arr = susceptibility(n_p, r, w_0, dn)
    chi = np.zeros((nz, nr))
    chi[:] = chi_arr

    # Preallocate array for spot size and propagation distance
    w = np.zeros(nt + 1)
    z = np.linspace(0, t_max * ct.c, nt + 1)

    # Initialize laser envelope
    laser.set_envelope_solver_params(zmin, zmax, rmax, nz, nr, dt, n_p)
    laser.initialize_envelope()

    # Calculate initial spot size
    w[0] = calculate_spot_size(laser.get_envelope(), dr)

    # Evolve laser
    for i in range(nt):
        laser.evolve(chi)
        w[i + 1] = calculate_spot_size(laser.get_envelope(), dr)

    # Plot spot size evolution
    zm = np.pi * w_0 ** 2 / l_0
    k_os = (2 / zm) * np.sqrt(dn / dn_c)
    w_theoretical = spotsize(dn_c, dn, w_0, w_0, k_os, z)
    max_err = np.max(np.abs(w - w_theoretical)) / w_0

    assert max_err < 0.04  # should be approx 0.0359
