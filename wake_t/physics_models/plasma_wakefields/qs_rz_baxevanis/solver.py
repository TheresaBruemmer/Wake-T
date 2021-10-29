"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
from numba import njit, vectorize, float64
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution
from wake_t.particles.interpolation import gather_sources_qs_baxevanis
from wake_t.utilities.other import radial_gradient


def calculate_wakefields(laser_a2, beam_part, r_max, xi_min, xi_max,
                         n_r, n_xi, ppc, n_p, r_max_plasma=None,
                         parabolic_coefficient=0., p_shape='cubic',
                         max_gamma=10.):
    """
    Calculate the plasma wakefields generated by the given laser pulse and
    electron beam in the specified grid points.

    Parameters:
    -----------
    laser_a2 : ndarray
        A (nz x nr) array containing the square of the laser envelope.

    beam_part : list
        List of numpy arrays containing the spatial coordinates and charge of
        all beam particles, i.e [x, y, xi, q].

    r_max : float
        Maximum radial position up to which plasma wakefield will be
        calculated.

    xi_min : float
        Minimum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.

    xi_max : float
        Maximum longitudinal (speed of light frame) position up to which
        plasma wakefield will be calculated.

    n_r : int
        Number of grid elements along r in which to calculate the wakefields.

    n_xi : int
        Number of grid elements along xi in which to calculate the wakefields.

    ppc : int (optional)
        Number of plasma particles per 1d cell along the radial direction.

    n_p : float
        Plasma density in units of m^{-3}.

    r_max_plasma : float
        Maximum radial extension of the plasma column. If `None`, the plasma
        extends up to the `r_max` boundary of the simulation box.

    parabolic_coefficient : float
        The coefficient for the transverse parabolic density profile. The
        radial density distribution is calculated as
        `n_r = n_p * (1 + parabolic_coefficient * r**2)`, where `n_p` is the
        local on-axis plasma density.

    p_shape : str
        Particle shape to be used for the beam charge deposition. Possible
        values are 'linear' or 'cubic'.

    max_gamma : float
        Plasma particles whose `gamma` exceeds `max_gamma` are considered to
        violate the quasistatic condition and are put at rest (i.e.,
        `gamma=1.`, `pr=pz=0.`).

    """
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d
    parabolic_coefficient = parabolic_coefficient * s_d**2

    # Initialize plasma particles.
    dr = r_max / n_r
    dr_p = dr / ppc
    # Maximum radial extent of the plasma.
    if r_max_plasma is None:
        r_max_plasma = r_max
    else:
        r_max_plasma = r_max_plasma / s_d
    n_part = int(np.round(r_max_plasma / dr * ppc))
    # Readjust plasma extent to match number of particles.
    r_max_plasma = n_part * dr_p
    r = np.linspace(dr_p / 2, r_max_plasma - dr_p / 2, n_part)
    pr = np.zeros_like(r)
    pz = np.zeros_like(r)
    gamma = np.ones_like(r)
    q = dr_p * r + dr_p * parabolic_coefficient * r**3

    # Iteration steps.
    dxi = (xi_max - xi_min) / (n_xi - 1)

    # Calculate and allocate laser quantities, including guard cells.
    a2_rz = np.zeros((n_xi+4, n_r+4))
    nabla_a2_rz = np.zeros((n_xi+4, n_r+4))
    a2_rz[2:-2, 2:-2] = laser_a2
    nabla_a2_rz[2:-2, 2:-2] = radial_gradient(laser_a2, dr)

    # Initialize field arrays, including guard cells.
    rho = np.zeros((n_xi+4, n_r+4))
    chi = np.zeros((n_xi+4, n_r+4))
    psi = np.zeros((n_xi+4, n_r+4))
    W_r = np.zeros((n_xi+4, n_r+4))
    E_z = np.zeros((n_xi+4, n_r+4))
    b_theta_bar = np.zeros((n_xi+4, n_r+4))

    # Field node coordinates.
    r_fld = np.linspace(dr / 2, r_max - dr / 2, n_r)
    xi_fld = np.linspace(xi_min, xi_max, n_xi)

    # Beam source. This code is needed while no proper support particle
    # beams as input is implemented.
    b_theta_0_mesh = calculate_beam_source_from_particles(
        *beam_part, n_p, n_r, n_xi, r_fld[0], xi_fld[0], dr, dxi, p_shape)

    # Main loop.
    for step in np.arange(n_xi):
        i = -1 - step
        xi = xi_fld[i]

        # Calculate source terms at position of plasma particles.
        a2, nabla_a2, b_theta_0 = gather_sources_qs_baxevanis(
            a2_rz, nabla_a2_rz, b_theta_0_mesh, xi_fld[0], xi_fld[-1],
            r_fld[0], r_fld[-1], dxi, dr, r, xi)

        # Get sorted particle indices
        idx = np.argsort(r)

        # Calculate wakefield potential and derivatives at plasma particles.
        out = calculate_psi_and_derivatives_at_particles(
            r, pr, q, idx, r_max_plasma, dr_p, parabolic_coefficient)
        psi_p, dr_psi_p, dxi_psi_p = out

        # Update gamma and pz of plasma particles
        gamma = calculate_gamma(pr, psi_p, a2)
        pz = calculate_pz(pr, psi_p, a2)

        # If particles violate the quasistatic condition, slow them down again.
        # This preserves the charge and shows better behavior than directly
        # removing them.
        idx_keep = np.where(gamma >= max_gamma)
        pz[idx_keep] = 0.
        gamma[idx_keep] = 1.
        pr[idx_keep] = 0.

        # Calculate fields at specified radii for current plasma column.
        calculate_psi(psi[i-2, 2:-2], r_fld, r, q, idx, r_max_plasma,
                      parabolic_coefficient)
        calculate_b_theta(b_theta_bar[i-2, 2:-2], r_fld, r, pr, q, gamma,
                          psi_p, dr_psi_p, dxi_psi_p, b_theta_0, nabla_a2, idx)

        # Deposit rho and chi of plasma column
        w_rho = q / (dr * r * (1 - pz/gamma))
        w_chi = w_rho / gamma
        z = np.full_like(r, xi)
        x = r
        y = np.zeros_like(r)
        deposit_3d_distribution(z, x, y, w_rho, xi_min, r_fld[0], n_xi, n_r,
                                dxi, dr, rho, p_shape=p_shape)
        deposit_3d_distribution(z, x, y, w_chi, xi_min, r_fld[0], n_xi, n_r,
                                dxi, dr, chi, p_shape=p_shape)

        if step < n_xi-1:
            # Evolve plasma to next xi step.
            evolve_plasma(
                r, pr, q, xi, dxi, dr_p, a2_rz, nabla_a2_rz, b_theta_0_mesh,
                xi_fld, r_fld, r_max_plasma, parabolic_coefficient)

    # Calculate derived fields (E_z, W_r, and E_r).
    dxi_psi, dr_psi = np.gradient(psi[2:-2, 2:-2], dxi, dr, edge_order=2)
    E_z[2:-2, 2:-2] = -dxi_psi
    W_r[2:-2, 2:-2] = -dr_psi
    # E_r = b_theta_bar + b_theta_0_mesh - W_r
    return rho, chi, W_r, E_z, xi_fld, r_fld


def evolve_plasma(r, pr, q, xi, dxi, dr_p, a2_rz, nabla_a2_rz, b_theta_0_mesh,
                  xi_fld, r_fld, r_max_plasma, pc):
    """
    Evolve the r and pr coordinates of plasma particles to the next xi step
    using a Runge-Kutta method of 4th order.

    This means that the transverse coordinates are updated as:
    r += (Ar + 2*Br + 2*Cr) + Dr) / 6
    pr += (Apr + 2*Bpr + 2*Cpr) + Dpr) / 6

    The required constants are calculated here and then passed to
    the jittable method 'update_particles_rk4' to apply the equations above.

    Parameters:
    -----------

    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        particles.

    xi : float
        Current xi position (speed-of-light frame) of the plasma particles.

    dxi : float
        Longitudinal step for the Runge-Kutta solver.

    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.

    a2_rz, nabla_a2_rz, b_theta_0_mesh : ndarray
        (nz+4, nr+4) arrays containing the source fields, i.e., the square
        of the laser envelope and its derivative as well as the azimuthal
        magnetic field of the beam species.

    xi_fld, r_fld : array
        Arrays containing the position of the field points.

    r_max_plasma : float
        Maximum radial extent of the plasma column.

    pc : float
        The parabolic density profile coefficient.

    """
    Ar, Apr = motion_derivatives(
        dxi, dr_p, xi, r, pr, q, a2_rz, nabla_a2_rz, b_theta_0_mesh, xi_fld,
        r_fld, r_max_plasma, pc)
    Br, Bpr = motion_derivatives(
        dxi, dr_p, xi - dxi / 2, r + Ar / 2, pr + Apr / 2, q, a2_rz,
        nabla_a2_rz, b_theta_0_mesh, xi_fld, r_fld, r_max_plasma, pc)
    Cr, Cpr = motion_derivatives(
        dxi, dr_p, xi - dxi / 2, r + Br / 2, pr + Bpr / 2, q, a2_rz,
        nabla_a2_rz, b_theta_0_mesh, xi_fld, r_fld, r_max_plasma, pc)
    Dr, Dpr = motion_derivatives(
        dxi, dr_p, xi - dxi, r + Cr, pr + Cpr, q, a2_rz, nabla_a2_rz,
        b_theta_0_mesh, xi_fld, r_fld, r_max_plasma, pc)
    return update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr)


def motion_derivatives(dxi, dr_p, xi, r, pr, q, a2_rz, nabla_a2_rz,
                       b_theta_0_mesh, xi_fld, r_fld, r_max_plasma, pc):
    """
    Return the derivatives of the radial position and momentum of the plasma
    particles.

    The method corrects for any particles with r < 0, calculates the source
    terms for the derivatives and delegates their calculation to the jittable
    method 'calculate_derivatives'.

    For details about the input parameters, check 'evolve_plasma' method.

    """
    # Check for particles with negative radial position. If so, invert them.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        # Make copy to avoid altering data for next Runge-Kutta step.
        r = r.copy()
        pr = pr.copy()
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.

    # Calculate source terms from laser and beam particles.
    xi_min = xi_fld[0]
    xi_max = xi_fld[-1]
    r_min = r_fld[0]
    r_max = r_fld[-1]
    dxi = xi_fld[1] - xi_fld[0]
    dr = r_fld[1] - r_fld[0]

    a2, nabla_a2, b_theta_0 = gather_sources_qs_baxevanis(
        a2_rz, nabla_a2_rz, b_theta_0_mesh, xi_min, xi_max, r_min, r_max,
        dxi, dr, r, xi)

    # Get sorted particle indices
    idx = np.argsort(r)

    # Calculate motion derivatives in jittable method.
    dr, dpr = calculate_derivatives(dxi, dr_p, r_max_plasma, r, pr, q,
                                    b_theta_0, nabla_a2, a2, idx, pc)

    # For particles which crossed the axis and where inverted, invert now
    # back the sign of the derivatives.
    if idx_neg[0].size > 0:
        dr[idx_neg] *= -1.
        dpr[idx_neg] *= -1.
    return dr, dpr


@njit()
def calculate_derivatives(dxi, dr_p, r_max, r, pr, q, b_theta_0, nabla_a2, a2,
                          idx, pc):
    """
    Jittable method to which the calculation of the motion derivatives is
    outsourced.

    Parameters:
    -----------
    dxi : float
        Longitudinal step for the Runge-Kutta solver.

    r, pr, q : ndarray
        Arrays containing the radial position, momentum and charge of the
        particles.

    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.

    r_max : float
        Maximum radial extent of the plasma column.

    b_theta_0 : ndarray
        Array containing the value of the azimuthal magnetic field from
        the beam distribution at the position of each particle.

    nabla_a2 : ndarray
        Array containing the value of the gradient of the laser normalized
        vector potential at the position of each particle.

    a2 : ndarray
        Array containing the value of the square of the laser normalized
        vector potential at the position of each particle.

    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.

    pc : float
        The parabolic density profile coefficient.

    """

    # Calculate wakefield potential and its derivaties at particle positions.
    psi, dr_psi, dxi_psi = calculate_psi_and_derivatives_at_particles(
        r, pr, q, idx, r_max, dr_p, pc)

    # Calculate gamma (Lorentz factor) of particles.
    gamma = calculate_gamma(pr, psi, a2)

    # Calculate azimuthal magnetic field from plasma at particle positions.
    b_theta_bar = calculate_b_theta_at_particles(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a2, idx, dr_p)

    # Calculate derivatives of r and pr.
    dr = calculate_dr(dxi, pr, psi)
    dpr = calculate_dpr(
        dxi, gamma, psi, dr_psi, b_theta_bar, b_theta_0, nabla_a2)
    return dr, dpr


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, float64)])
def calculate_dpr(dxi, gamma, psi, dr_psi, b_theta_bar, b_theta_0, nabla_a2):
    return dxi * (gamma * dr_psi / (1. + psi)
                        - b_theta_bar
                        - b_theta_0
                        - nabla_a2 / (2. * (1. + psi)))


@vectorize([float64(float64, float64, float64)])
def calculate_dr(dxi, pr, psi):
    return dxi * pr / (1. + psi)


@vectorize([float64(float64, float64, float64)])
def calculate_gamma(pr, psi, a2):
    return (1. + pr ** 2 + a2 + (1. + psi) ** 2) / (2. * (1. + psi))


@vectorize([float64(float64, float64, float64)])
def calculate_pz(pr, psi, a2):
    return (1. + pr ** 2 + a2 - (1. + psi) ** 2) / (2. * (1. + psi))


@njit()
def update_particles_rk4(r, pr, Ar, Br, Cr, Dr, Apr, Bpr, Cpr, Dpr):
    """
    Jittable method to which updating the particle coordinates in the RK4
    algorithm is outsourced.

    It also checks and corrects for any particles with r < 0.

    """
    # Push particles
    inv_6 = 1. / 6.
    for i in range(r.shape[0]):
        r[i] += (Ar[i] + 2. * (Br[i] + Cr[i]) + Dr[i]) * inv_6
        pr[i] += (Apr[i] + 2. * (Bpr[i] + Cpr[i]) + Dpr[i]) * inv_6
    # Check if any have a negative radial position. If so, invert them.
    idx_neg = np.where(r < 0.)
    if idx_neg[0].size > 0:
        r[idx_neg] *= -1.
        pr[idx_neg] *= -1.
    return


@njit
def update_gamma_and_pz(gamma, pz, pr, a2, psi):
    """
    Update the gamma factor and longitudinal momentum of the plasma particles.

    Parameters:
    -----------
    gamma, pz : ndarray
        Arrays containing the current gamma factor and longitudinal momentum
        of the plasma particles (will be modified here).

    pr, a2, psi : ndarray
        Arrays containing the radial momentum of the particles and the
        value of a2 and psi at the position of the particles.

    """
    for i in range(pr.shape[0]):
        gamma[i] = (1 + pr[i]**2 + a2[i] + (1+psi[i])**2) / (2 * (1+psi[i]))
        pz[i] = (1 + pr[i]**2 + a2[i] - (1+psi[i])**2) / (2 * (1+psi[i]))


@njit()
def calculate_psi_and_derivatives_at_particles(r, pr, q, idx, r_max, dr_p, pc):
    """
    Calculate the wakefield potential and its derivatives at the position
    of the plasma particles. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters:
    -----------
    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.

    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.

    r_max : float
        Maximum radial extent of the plasma column.

    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.

    """
    # Initialize arrays.
    n_part = r.shape[0]
    psi = np.empty(n_part)
    dr_psi = np.empty(n_part)
    dxi_psi = np.empty(n_part)

    # Initialize value of sums.
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate psi and dr_psi.
    # Their value at the position of each plasma particle is calculated
    # by doing a linear interpolation between two values at the left and
    # right of the particle. The left point is the middle position between the
    # particle and its closest left neighbor, and the same for the right.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        # Calculate new sums.
        sum_1_new = sum_1 + q_i
        sum_2_new = sum_2 + q_i * np.log(r_i)

        # If this is not the first particle, calculate the left point (r_left)
        # and the field values there (psi_left and dr_psi_left) as usual.
        if i_sort > 0:
            r_im1 = r[idx[i_sort-1]]
            r_left = (r_im1 + r_i) / 2
            psi_left = delta_psi_eq(r_left, sum_1, sum_2, r_max, pc)
            dr_psi_left = dr_psi_eq(r_left, sum_1, r_max, pc)
        # Otherwise, take r=0 as the location of the left point.
        else:
            r_left = 0.
            psi_left = 0.
            dr_psi_left = 0.

        # If this is not the last particle, calculate the r_right as
        # middle point.
        if i_sort < n_part - 1:
            r_ip1 = r[idx[i_sort+1]]
            r_right = (r_i + r_ip1) / 2
        # Otherwise, since the particle represents a charge sheet of width
        # dr_p, take the right point as r_i + dr_p/2.
        else:
            r_right = r_i + dr_p/2
        # Calculate field values ar r_right.
        psi_right = delta_psi_eq(r_right, sum_1_new, sum_2_new, r_max, pc)
        dr_psi_right = dr_psi_eq(r_right, sum_1_new, r_max, pc)

        # Interpolate psi.
        b_1 = (psi_right - psi_left) / (r_right - r_left)
        a_1 = psi_left - b_1*r_left
        psi[i] = a_1 + b_1*r_i

        # Interpolate dr_psi.
        b_2 = (dr_psi_right - dr_psi_left) / (r_right - r_left)
        a_2 = dr_psi_left - b_2*r_left
        dr_psi[i] = a_2 + b_2*r_i

        # Update value of sums.
        sum_1 = sum_1_new
        sum_2 = sum_2_new

    # Boundary condition for psi (Force potential to be zero either at the
    # plasma edge or after the last particle, whichever is further away).
    r_furthest = max(r_right, r_max)
    psi -= delta_psi_eq(r_furthest, sum_1, sum_2, r_max, pc)

    # In theory, psi cannot be smaller than -1. However, it has been observed
    # than in very strong blowouts, near the peak, values below -1 can appear
    # in this numerical method. In addition, values very close to -1 will lead
    # to particles with gamma >> 10, which will also lead to problems.
    # This condition here makes sure that this does not happen, improving
    # the stability of the solver.
    for i in range(n_part):
        # Should only happen close to the peak of very strong blowouts.
        if psi[i] < -0.90:
            psi[i] = -0.90

    # Calculate dxi_psi (also by interpolation).
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi[i]

        sum_3_new = sum_3 + (q_i * pr_i) / (r_i * (1 + psi_i))

        # Check if it is the first particle.
        if i_sort > 0:
            r_im1 = r[idx[i_sort-1]]
            r_left = (r_im1 + r_i) / 2
            dxi_psi_left = -sum_3
        else:
            r_left = 0.
            dxi_psi_left = 0.

        # Check if it is the last particle.
        if i_sort < n_part - 1:
            r_ip1 = r[idx[i_sort+1]]
            r_right = (r_i + r_ip1) / 2
        else:
            r_right = r_i + dr_p/2
        dxi_psi_right = -sum_3_new

        # Do interpolation.
        b = (dxi_psi_right - dxi_psi_left) / (r_right - r_left)
        a = dxi_psi_left - b*r_left
        dxi_psi[i] = a + b*r_i
        sum_3 = sum_3_new

    # Apply longitudinal derivative of the boundary conditions of psi.
    if r_right <= r_max:
        dxi_psi += sum_3
    else:
        dxi_psi += sum_3 - ((sum_1 - r_max**2/2 - pc*r_max/4)
                            * pr_i / (r_right * (1 + psi_i)))

    # Again, near the peak of a strong blowout, very large and unphysical
    # values could appear. This condition makes sure a threshold us not
    # exceeded.
    for i in range(n_part):
        if dxi_psi[i] > 3.:
            dxi_psi[i] = 3.
        elif dxi_psi[i] < -3.:
            dxi_psi[i] = -3.

    return psi, dr_psi, dxi_psi


@njit()
def calculate_psi(psi, r_fld, r, q, idx, r_max, pc):
    """
    Calculate the wakefield potential at the radial
    positions specified in r_fld. This is done by using Eq. (29) in
    the paper by P. Baxevanis and G. Stupakov.

    Parameters:
    -----------
    psi : array
        Array into which the calculated `psi` will be stored.

    r_fld : array
        Array containing the radial positions where psi should be calculated.
        Has same dimensions as `psi`.

    r, q : array
        Arrays containing the radial position, and charge of the
        plasma particles.

    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.

    r_max : float
        Maximum radial extent of the plasma column.

    pc : float
        The parabolic density profile coefficient.

    """
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]
    sum_1_arr = np.empty(n_part)
    sum_2_arr = np.empty(n_part)
    sum_1 = 0.
    sum_2 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
    r_N = r_i

    # Calculate fields at r_fld.
    i_last = 0
    for j in range(r_fld.shape[0]):
        r_j = r_fld[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            sum_1_j = 0.
            sum_2_j = 0.
            i_last = 0
        else:
            i = idx[i_last]
            sum_1_j = sum_1_arr[i]
            sum_2_j = sum_2_arr[i]
        psi[j] = delta_psi_eq(r_j, sum_1_j, sum_2_j, r_max, pc)

    # Apply boundary conditions.
    r_furthest = max(r_N, r_max)
    psi -= delta_psi_eq(r_furthest, sum_1, sum_2, r_max, pc)


@njit()
def delta_psi_eq(r, sum_1, sum_2, r_max, pc):
    """ Adapted equation (29) from original paper. """
    delta_psi_elec = sum_1*np.log(r) - sum_2
    if r <= r_max:
        delta_psi_ion = 0.25*r**2 + pc*r**4/16
    else:
        delta_psi_ion = (
            0.25*r_max**2 + pc*r_max**4/16 +
            (0.5 * r_max**2 + 0.25*pc*r_max**4) * (
                np.log(r)-np.log(r_max)))
    return delta_psi_elec - delta_psi_ion


@njit()
def dr_psi_eq(r, sum_1, r_max, pc):
    """ Adapted equation (31) from original paper. """
    dr_psi_elec = sum_1 / r
    if r <= r_max:
        dr_psi_ion = 0.5 * r + 0.25 * pc * r ** 3
    else:
        dr_psi_ion = (0.5 * r_max**2 + 0.25 * pc * r_max**4) / r
    return dr_psi_elec - dr_psi_ion


@njit()
def calculate_psi_and_derivatives(r_fld, r, pr, q):
    """
    Calculate the wakefield potential and its derivatives at the radial
    positions specified in r_fld. This is done by using Eqs. (29) - (32) in
    the paper by P. Baxevanis and G. Stupakov.

    Parameters:
    -----------
    r_fld : array
        Array containing the radial positions where psi should be calculated.

    r, pr, q : array
        Arrays containing the radial position, momentum and charge of the
        plasma particles.

    """
    # Initialize arrays with values of psi and sums at plasma particles.
    n_part = r.shape[0]
    psi_part = np.zeros(n_part)
    sum_1_arr = np.zeros(n_part)
    sum_2_arr = np.zeros(n_part)
    sum_3_arr = np.zeros(n_part)
    sum_1 = 0.
    sum_2 = 0.
    sum_3 = 0.

    # Calculate sum_1, sum_2 and psi_part.
    idx = np.argsort(r)
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]

        sum_1 += q_i
        sum_2 += q_i * np.log(r_i)
        sum_1_arr[i] = sum_1
        sum_2_arr[i] = sum_2
        psi_part[i] = sum_1 * np.log(r_i) - sum_2 - 0.25 * r_i ** 2
    r_N = r[-1]
    psi_part += - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)

    # Calculate sum_3.
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        psi_i = psi_part[i]

        sum_3 += (q_i * pr_i) / (r_i * (1 + psi_i))
        sum_3_arr[i] = sum_3

    # Initialize arrays for psi and derivatives at r_fld locations.
    n_points = r_fld.shape[0]
    psi = np.zeros(n_points)
    dr_psi = np.zeros(n_points)
    dxi_psi = np.zeros(n_points)

    # Calculate fields at r_fld.
    i_last = 0
    for j in range(n_points):
        r_j = r_fld[j]
        # Get index of last plasma particle with r_i < r_j.
        for i_sort in range(n_part):
            i = idx[i_sort]
            r_i = r[i]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields at r_j.
        if i_last == -1:
            psi[j] = -0.25 * r_j ** 2
            dr_psi[j] = -0.5 * r_j
            dxi_psi[j] = 0.
        else:
            i_p = idx[i_last]
            psi[j] = sum_1_arr[i_p] * np.log(r_j) - sum_2_arr[
                i_p] - 0.25 * r_j ** 2
            dr_psi[j] = sum_1_arr[i_p] / r_j - 0.5 * r_j
            dxi_psi[j] = - sum_3_arr[i_p]
    psi = psi - (sum_1 * np.log(r_N) - sum_2 - 0.25 * r_N ** 2)
    dxi_psi = dxi_psi + sum_3
    return psi, dr_psi, dxi_psi


@njit()
def calculate_b_theta_at_particles(r, pr, q, gamma, psi, dr_psi, dxi_psi,
                                   b_theta_0, nabla_a2, idx, dr_p):
    """
    Calculate the azimuthal magnetic field from the plasma at the location
    of the plasma particles using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    As indicated in the original paper, the value of the fields at the
    discontinuities (at the exact radial position of the plasma particles)
    is calculated as the average between the two neighboring values.

    Parameters:
    -----------
    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.

    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.

    b_theta_0, nabla_a2 : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.

    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.

    dr_p : float
        Initial spacing between plasma macroparticles. Corresponds also the
        width of the plasma sheet represented by the macroparticle.

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0 = calculate_ai_bi_from_edge(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a2, idx)

    # Calculate field at particles as average between neighboring values.
    n_part = r.shape[0]

    # Preallocate field array.
    b_theta_bar = np.empty(n_part)

    # Calculate field value at plasma particles by interpolating between two
    # neighboring values. Same as with psi and its derivaties.
    for i_sort in range(n_part):
        i = idx[i_sort]
        im1 = idx[i_sort-1]
        ip1 = idx[i_sort+1]
        r_i = r[i]
        if i_sort > 0:
            r_im1 = r[im1]
            a_im1 = a_i[im1]
            b_im1 = b_i[im1]
            r_left = (r_im1 + r_i) / 2
            b_theta_left = a_im1 * r_left + b_im1 / r_left
        else:
            b_theta_left = 0.
            r_left = 0.
        if i_sort < n_part - 1:
            r_ip1 = r[ip1]
            r_right = (r_i + r_ip1) / 2
        else:
            r_right = r_i + dr_p / 2
        b_theta_right = a_i[i] * r_right + b_i[i] / r_right

        # Do interpolation.
        b = (b_theta_right - b_theta_left) / (r_right - r_left)
        a = b_theta_left - b*r_left
        b_theta_bar[i] = a + b*r_i

        # Near the peak of a strong blowout, very large and unphysical
        # values could appear. This condition makes sure a threshold us not
        # exceeded.
        if b_theta_bar[i] > 3.:
            b_theta_bar[i] = 3.
        elif b_theta_bar[i] < -3.:
            b_theta_bar[i] = -3.

    return b_theta_bar


@njit()
def calculate_b_theta(
        b_theta_mesh, r_fld, r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
        nabla_a2, idx):
    """
    Calculate the azimuthal magnetic field from the plasma at the radial
    locations in r_fld using Eqs. (24), (26) and (27) from the paper
    of P. Baxevanis and G. Stupakov.

    Parameters:
    -----------
    b_theta_mesh : array
        Array into which the calculated `b_theta` will be stored.

    r_fld : array
        Array containing the radial positions where psi should be calculated.
        Has same dimensions as `b_theta_mesh`.

    r, pr, q, gamma : arrays
        Arrays containing, respectively, the radial position, radial momentum,
        charge and gamma (Lorentz) factor of the plasma particles.

    psi, dr_psi, dxi_psi : arrays
        Arrays with the value of the wakefield potential and its radial and
        longitudinal derivatives at the location of the plasma particles.

    b_theta_0, nabla_a2 : arrays
        Arrays with the value of the source terms. The first one being the
        azimuthal magnetic field due to the beam distribution, and the second
        the gradient of the normalized vector potential of the laser.

    idx : ndarray
        Array containing the (radially) sorted indices of the plasma particles.

    """
    # Calculate a_i and b_i, as well as a_0 and the sorted particle indices.
    a_i, b_i, a_0 = calculate_ai_bi_from_edge(
        r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0, nabla_a2, idx)

    i_last = 0
    for j in range(r_fld.shape[0]):
        r_j = r_fld[j]
        # Get index of last plasma particle with r_i < r_j, continuing from
        # last particle found in previous iteration.
        for i_sort in range(i_last, r.shape[0]):
            i_p = idx[i_sort]
            r_i = r[i_p]
            i_last = i_sort
            if r_i >= r_j:
                i_last -= 1
                break
        # Calculate fields.
        if i_last == -1:
            b_theta_mesh[j] = a_0 * r_j
            i_last = 0
        else:
            i_p = idx[i_last]
            b_theta_mesh[j] = a_i[i_p] * r_j + b_i[i_p] / r_j


@njit()
def calculate_ai_bi_from_axis(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                              nabla_a2, idx):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters see method 'calculate_b_theta'.

    The values of a_i and b_i are calculated as follows, using Eqs. (26) and
    (27) from the paper of P. Baxevanis and G. Stupakov:

        Write a_i and b_i as linear system of a_0:

            a_i = K_i * a_0 + T_i
            b_i = U_i * a_0 + P_i


        Where (im1 stands for subindex i-1):

            K_i = (1 + A_i*r_i/2) * K_im1  +  A_i/(2*r_i)     * U_im1
            U_i = (-A_i*r_i**3/2) * K_im1  +  (1 - A_i*r_i/2) * U_im1

            T_i = ( (1 + A_i*r_i/2) * T_im1  +  A_i/(2*r_i)     * P_im1  +
                    (2*Bi + Ai*Ci)/4 )
            P_i = ( (-A_i*r_i**3/2) * T_im1  +  (1 - A_i*r_i/2) * P_im1  +
                    r_i*(4*Ci - 2*Bi*r_i - Ai*Ci*r_i)/4 )

        With initial conditions:

            K_0 = 1
            U_0 = 0
            T_0 = 0
            P_0 = 0

        Then a_0 can be determined by imposing a_N = 0:

            a_N = K_N * a_0 + T_N = 0 <=> a_0 = - T_N / K_N

    """
    n_part = r.shape[0]

    # Preallocate arrays
    K = np.empty(n_part)
    U = np.empty(n_part)
    T = np.empty(n_part)
    P = np.empty(n_part)

    # Establish initial conditions (K_0 = 1, U_0 = 0, O_0 = 0, P_0 = 0)
    K_im1 = 1.
    U_im1 = 0.
    T_im1 = 0.
    P_im1 = 0.

    # Iterate over particles
    for i_sort in range(n_part):
        i = idx[i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        gamma_i = gamma[i]
        psi_i = psi[i]
        dr_psi_i = dr_psi[i]
        dxi_psi_i = dxi_psi[i]
        b_theta_0_i = b_theta_0[i]
        nabla_a2_i = nabla_a2[i]

        a = 1. + psi_i
        a2 = a * a
        a3 = a2 * a
        b = 1. / (r_i * a)
        c = 1. / (r_i * a2)
        pr_i2 = pr_i * pr_i

        A_i = q_i * b
        B_i = q_i * (- (gamma_i * dr_psi_i) * c
                     + (pr_i2 * dr_psi_i) / (r_i * a3)
                     + (pr_i * dxi_psi_i) * c
                     + pr_i2 / (r_i * r_i * a2)
                     + b_theta_0_i * b
                     + nabla_a2_i * c * 0.5)
        C_i = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)

        l_i = (1. + 0.5 * A_i * r_i)
        m_i = 0.5 * A_i / r_i
        n_i = -0.5 * A_i * r_i ** 3
        o_i = (1. - 0.5 * A_i * r_i)

        K_i = l_i * K_im1 + m_i * U_im1
        U_i = n_i * K_im1 + o_i * U_im1
        T_i = l_i * T_im1 + m_i * P_im1 + 0.5 * B_i + 0.25 * A_i * C_i
        P_i = n_i * T_im1 + o_i * P_im1 + r_i * (
                C_i - 0.5 * B_i * r_i - 0.25 * A_i * C_i * r_i)

        K[i] = K_i
        U[i] = U_i
        T[i] = T_i
        P[i] = P_i

        K_im1 = K_i
        U_im1 = U_i
        T_im1 = T_i
        P_im1 = P_i

    # Calculate a_0.
    a_0 = - T_im1 / K_im1

    # Calculate a_i and b_i as functions of a_0.
    a_i = K * a_0 + T
    b_i = U * a_0 + P
    return a_i, b_i, a_0


@njit()
def calculate_ai_bi_from_edge(r, pr, q, gamma, psi, dr_psi, dxi_psi, b_theta_0,
                              nabla_a2, idx):
    """
    Calculate the values of a_i and b_i which are needed to determine
    b_theta at any r position.

    For details about the input parameters see method 'calculate_b_theta'.

    The values of a_i and b_i are calculated, using Eqs. (26) and
    (27) from the paper of P. Baxevanis and G. Stupakov. In this algorithm,
    Eq. (27) is inverted so that we calculate a_im1 and b_im1 as a function
    of a_i and b_i. Therefore, we start the loop at the boundary and end up
    on axis. This alternative method has shown to be more robust than
    `calculate_ai_bi_from_axis` to numerical precission issues.

        Write a_im1 and b_im1 as linear system of b_N:

            a_im1 = K_i * b_N + T_i
            b_im1 = U_i * b_N + P_i


        Where (im1 stands for subindex i-1):

            K_i = (1 - A_i*r_i/2) * K_im1  +  (-A_i/(2*r_i))  * U_im1
            U_i = A_i*r_i**3/2    * K_im1  +  (1 + A_i*r_i/2) * U_im1

            T_i = ( (1 - A_i*r_i/2) * T_im1  +  (-A_i/(2*r_i))  * P_im1  +
                    (-2*Bi + Ai*Ci)/4 )
            P_i = ( A_i*r_i**3/2    * T_im1  +  (1 + A_i*r_i/2) * P_im1  -
                    r_i*(4*Ci - 2*Bi*r_i + Ai*Ci*r_i)/4 )

        With initial conditions at i=N+1:

            K_Np1 = 0
            U_Np1 = 1
            T_Np1 = 0
            P_Np1 = 0

        Then b_N can be determined by imposing b_0 = 0:

            b_0 = K_1 * b_N + T_1 = 0 <=> b_N = - T_1 / K_1

    """
    n_part = r.shape[0]

    # Preallocate arrays
    K = np.empty(n_part+1)
    U = np.empty(n_part+1)
    T = np.empty(n_part+1)
    P = np.empty(n_part+1)

    # Initial conditions at i = N+1
    K_ip1 = 0.
    U_ip1 = 1.
    T_ip1 = 0.
    P_ip1 = 0.
    K[-1] = K_ip1
    U[-1] = U_ip1
    T[-1] = T_ip1
    P[-1] = P_ip1

    # Sort particles

    # Iterate over particles
    for i_sort in range(n_part):
        i = idx[-1-i_sort]
        r_i = r[i]
        pr_i = pr[i]
        q_i = q[i]
        gamma_i = gamma[i]
        psi_i = psi[i]
        dr_psi_i = dr_psi[i]
        dxi_psi_i = dxi_psi[i]
        b_theta_0_i = b_theta_0[i]
        nabla_a2_i = nabla_a2[i]

        a = 1. + psi_i
        a2 = a * a
        a3 = a2 * a
        b = 1. / (r_i * a)
        c = 1. / (r_i * a2)
        pr_i2 = pr_i * pr_i

        A_i = q_i * b
        B_i = q_i * (- (gamma_i * dr_psi_i) * c
                     + (pr_i2 * dr_psi_i) / (r_i * a3)
                     + (pr_i * dxi_psi_i) * c
                     + pr_i2 / (r_i * r_i * a2)
                     + b_theta_0_i * b
                     + nabla_a2_i * c * 0.5)
        C_i = q_i * (pr_i2 * c - (gamma_i / a - 1.) / r_i)

        l_i = (1. - 0.5 * q_i / a)
        m_i = -0.5 * q_i / (a*r_i**2)
        n_i = 0.5 * q_i/a * r_i ** 2
        o_i = (1. + 0.5 * q_i / a)

        K_i = l_i * K_ip1 + m_i * U_ip1
        U_i = n_i * K_ip1 + o_i * U_ip1
        T_i = l_i * T_ip1 + m_i * P_ip1 - 0.5 * B_i + 0.25 * A_i * C_i
        P_i = n_i * T_ip1 + o_i * P_ip1 - r_i * (
                C_i - 0.5 * B_i * r_i + 0.25 * A_i * C_i * r_i)

        K[i] = K_i
        U[i] = U_i
        T[i] = T_i
        P[i] = P_i

        K_ip1 = K_i
        U_ip1 = U_i
        T_ip1 = T_i
        P_ip1 = P_i

    # Calculate b_N.
    b_N = - P_ip1 / U_ip1

    # Calculate a_i and b_i as functions of b_N.
    a_i = K * b_N + T
    b_i = U * b_N + P

    # Get a_0 (value on-axis) and make sure a_i and b_i only contain the values
    # at the plasma particles.
    a_0 = a_i[idx[0]]
    a_i = np.delete(a_i, idx[0])
    b_i = np.delete(b_i, idx[0])

    return a_i, b_i, a_0


def calculate_beam_source_from_particles(
        x, y, xi, q, n_p, n_r, n_xi, r_min, xi_min, dr, dxi, p_shape):
    """
    Return a (nz+4, nr+4) array with the azimuthal magnetic field
    from a particle distribution. This is Eq. (18) in the original paper.

    """
    # Plasma skin depth.
    s_d = ge.plasma_skin_depth(n_p / 1e6)

    # Get and normalize particle coordinate arrays.
    xi_n = xi / s_d
    x_n = x / s_d
    y_n = y / s_d

    # Calculate particle weights.
    w = - q / ct.e / (2 * np.pi * dr * dxi * s_d ** 3 * n_p)

    # Obtain charge distribution (using cubic particle shape by default).
    q_dist = np.zeros((n_xi + 4, n_r + 4))
    deposit_3d_distribution(xi_n, x_n, y_n, w, xi_min, r_min, n_xi, n_r, dxi,
                            dr, q_dist, p_shape=p_shape, use_ruyten=True)

    # Remove guard cells.
    q_dist = q_dist[2:-2, 2:-2]

    # Allovate magnetic field array.
    b_theta = np.zeros((n_xi+4, n_r+4))

    # Radial position of grid points.
    r_grid_g = (0.5 + np.arange(n_r)) * dr

    # At each grid cell, calculate integral only until cell center by
    # assuming that half the charge is evenly distributed within the cell
    # (i.e., substract half the charge)
    subs = q_dist / 2

    # At the first grid point along r, subtstact an additonal 1/4 of the
    # charge. This comes from assuming that the density has to be zero on axis.
    subs[:, 0] += q_dist[:, 0]/4

    # Calculate field by integration.
    b_theta[2:-2, 2:-2] = (
        (np.cumsum(q_dist, axis=1) - subs) * dr / np.abs(r_grid_g))

    return b_theta
