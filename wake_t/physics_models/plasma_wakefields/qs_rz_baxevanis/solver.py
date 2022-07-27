"""
This module implements the methods for calculating the plasma wakefields
using the 2D r-z reduced model from P. Baxevanis and G. Stupakov.

See https://journals.aps.org/prab/abstract/10.1103/PhysRevAccelBeams.21.071301
for the full details about this model.
"""

import numpy as np
import scipy.constants as ct
import aptools.plasma_accel.general_equations as ge

from wake_t.particles.deposition import deposit_3d_distribution
from .deposition import deposit_plasma_particles
from wake_t.particles.interpolation import gather_sources_qs_baxevanis
from wake_t.utilities.other import radial_gradient
from .plasma_push.rk4 import evolve_plasma_rk4
from .plasma_push.ab5 import evolve_plasma_ab5
from .psi_and_derivatives import (
    calculate_psi, calculate_psi_and_derivatives_at_particles)
from .b_theta import calculate_b_theta, calculate_b_theta_at_particles
from .plasma_particles import PlasmaParticles
from wake_t.utilities.numba import njit_serial


def calculate_wakefields(laser_a2, bunches, r_max, xi_min, xi_max,
                         n_r, n_xi, ppc, n_p, r_max_plasma=None,
                         parabolic_coefficient=0., p_shape='cubic',
                         max_gamma=10., plasma_pusher='rk4'):
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

    plasma_pusher : str
        Numerical pusher for the plasma particles. Possible values are `'rk4'`
        and `'ab5'`.

    """
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    r_max = r_max / s_d
    xi_min = xi_min / s_d
    xi_max = xi_max / s_d
    dr = r_max / n_r
    dxi = (xi_max - xi_min) / (n_xi - 1)
    parabolic_coefficient = parabolic_coefficient * s_d**2

    # Maximum radial extent of the plasma.
    if r_max_plasma is None:
        r_max_plasma = r_max
    else:
        r_max_plasma = r_max_plasma / s_d

    # Initialize plasma particles.
    pp = PlasmaParticles(
        r_max, r_max_plasma, parabolic_coefficient, dr, ppc, plasma_pusher)
    pp.initialize()

    # Initialize field arrays, including guard cells.
    a2 = np.zeros((n_xi+4, n_r+4))
    nabla_a2 = np.zeros((n_xi+4, n_r+4))
    rho = np.zeros((n_xi+4, n_r+4))
    chi = np.zeros((n_xi+4, n_r+4))
    psi = np.zeros((n_xi+4, n_r+4))
    W_r = np.zeros((n_xi+4, n_r+4))
    E_z = np.zeros((n_xi+4, n_r+4))
    b_t_bar = np.zeros((n_xi+4, n_r+4))

    # Field node coordinates.
    r_fld = np.linspace(dr / 2, r_max - dr / 2, n_r)
    xi_fld = np.linspace(xi_min, xi_max, n_xi)

    # Laser source.
    a2[2:-2, 2:-2] = laser_a2
    nabla_a2[2:-2, 2:-2] = radial_gradient(laser_a2, dr)

    # Beam source. This code is needed while no proper support particle
    # beams as input is implemented.
    b_t_beam = np.zeros((n_xi+4, n_r+4))
    for bunch in bunches:
        calculate_beam_source(bunch, n_p, n_r, n_xi, r_fld[0], xi_fld[0],
                              dr, dxi, p_shape, b_t_beam)

    # Evolve plasma from right to left and calculate psi, b_t_bar, rho and
    # chi on a grid.
    evolve_plasma_and_calculate_fields(
        pp, a2, nabla_a2, b_t_beam, psi, b_t_bar, rho, chi,
        xi_fld, r_fld, dxi, dr, n_xi, n_r,
        max_gamma, p_shape, plasma_pusher)

    # Calculate derived fields (E_z, W_r, and E_r).
    dxi_psi, dr_psi = np.gradient(psi[2:-2, 2:-2], dxi, dr, edge_order=2)
    E_z[2:-2, 2:-2] = -dxi_psi
    W_r[2:-2, 2:-2] = -dr_psi
    B_theta = b_t_bar + b_t_beam
    E_r = W_r + B_theta
    return rho, chi, E_r, E_z, B_theta, xi_fld, r_fld


def evolve_plasma_and_calculate_fields(
        pp, a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
        xi_fld, r_fld, dxi, dr, n_xi, n_r,
        max_gamma, p_shape, plasma_pusher):
    """Evolve plasma column from right to left and calculate plasma fields.

    Parameters
    ----------
    pp : PlasmaParticles
        The column of plasma particles to be evolved.
    a2, nabla_a2, b_t_0 : ndarray
        Arrays containing the laser and beam sources.
    psi, b_t_bar, rhi, chi : ndarray
        Arrays to be filled in while evolving the plasma column. They contain,
        respectively, the wakefield potential, the azimuthal magnetic field
        from the plasma, the plasma charge density and the plasma
        susceptibility.
    xi_fld, r_fld : ndarray
        Array containing the longitudinal and radial coordinates of the
        grid points.
    dxi, dr : float
        Longitudinal and radial step size.
    n_xi, n_r : int
        Number of grid elements in the longitudinal and radial direction.
    max_gamma : float
        Maximum gamma allowed for the plasma particles.
    p_shape : str
        Particle shape.
    plasma_pusher : str
        The plasma particle pusher.
    """

    # Calculate plasma evolution with Adams-Bashforth pusher.
    if plasma_pusher == 'ab5':
        dr_arrays, dpr_arrays = pp.get_ab5_arrays()
        calculate_with_ab5(
            pp.r, pp.pr, pp.pz, pp.gamma, pp.q,
            pp.r_max_plasma, pp.dr_p, pp.parabolic_coefficient,
            *pp.get_field_arrays(),
            a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
            xi_fld, r_fld, dxi, dr, n_xi, n_r,
            max_gamma, p_shape,
            *dr_arrays, *dpr_arrays
        )

    # Calculate plasma evolution with Runge-Kutta pusher.
    elif plasma_pusher == 'rk4':
        dr_arrays, dpr_arrays = pp.get_rk4_arrays()
        calculate_with_rk4(
            pp.r, pp.pr, pp.pz, pp.gamma, pp.q,
            pp.r_max_plasma, pp.dr_p, pp.parabolic_coefficient,
            a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
            xi_fld, r_fld, dxi, dr, n_xi, n_r,
            max_gamma, p_shape,
            *dr_arrays, *dpr_arrays,
            *pp.get_rk4_field_arrays(0),
            *pp.get_rk4_field_arrays(1),
            *pp.get_rk4_field_arrays(2),
            *pp.get_rk4_field_arrays(3)
        )

    # Raise error if pusher is not recognized.
    else:
        raise ValueError(
            "Plasma pusher '{}' not recognized.".format(plasma_pusher))


@njit_serial()
def calculate_with_ab5(
        r_pp, pr_pp, pz_pp, gamma_pp, q_pp,
        r_max_plasma, dr_p, parabolic_coefficient,
        a2_pp, nabla_a2_pp, b_t_0_pp, b_t_pp, psi_pp, dr_psi_pp, dxi_psi_pp,
        a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
        xi_fld, r_fld, dxi, dr, n_xi, n_r,
        max_gamma, p_shape,
        dr_1, dr_2, dr_3, dr_4, dr_5, dpr_1, dpr_2, dpr_3, dpr_4, dpr_5):
    """Calculate plasma evolution using the Adams-Bashforth pusher.

    Parameters
    ----------
    r_pp, pr_pp, pz_pp, gamma_pp, q_pp : ndarray
        Radial position, radial momentum, longitudinal momentum,
        Lorentz factor and charge of the plasma particles.
    r_max_plasma : float
        Maximum radial extent of the plasma
    dr_p : float
        Initial radial spacing between plasma particles.
    parabolic_coefficient : float
        Coefficient for the parabolic radial plasma profile.
    a2_pp, ..., dxi_psi_pp : ndarray
        Arrays where the value of the fields at the particle positions will
        be stored.
    a2, nabla_a2, b_t_0 : ndarray
        Laser and beam source fields.
    psi, b_t_bar, rho, chi : ndarray
        Arrays to be filled in during plasma evolution.
    xi_fld, r_fld : ndarray
        Grid coordinates
    dxi, dr : float
        Grid spacing
    n_xi, n_r : int
        Number of grid elements.
    max_gamma : float
        Maximum gamma of the plasma particles.
    p_shape : str
        Particle shape.
    dr_1, ..., dr_5 : ndarray
        Arrays containing the derivative of the radial position of the
        particles at the 5 slices previous to the next one.
    dpr_1, ..., dpr_5 : ndarray
        Arrays containing the derivative of the radial momentum of the
        particles at the 5 slices previous to the next one.
    """
    # Loop from the right to the left of the domain.
    for step in range(n_xi):
        slice_i = n_xi - step - 1
        xi = xi_fld[slice_i]

        # Calculate fields at the position of the particles and
        # calculate/deposit psi, b_t_bar, rho and chi at the current slice
        # of the grid.
        calculate_and_deposit_plasma_column(
            slice_i, xi, r_pp, pr_pp, pz_pp, gamma_pp, q_pp,
            r_max_plasma, dr_p, parabolic_coefficient,
            a2_pp, nabla_a2_pp, b_t_0_pp, b_t_pp,
            psi_pp, dr_psi_pp, dxi_psi_pp,
            a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
            xi_fld, r_fld, dxi, dr, n_xi, n_r,
            max_gamma, p_shape)

        if slice_i > 0:
            # Evolve plasma to next xi step.
            evolve_plasma_ab5(
                dxi, r_pp, pr_pp, gamma_pp,
                nabla_a2_pp, b_t_0_pp, b_t_pp, psi_pp, dr_psi_pp,
                dr_1, dr_2, dr_3, dr_4, dr_5,
                dpr_1, dpr_2, dpr_3, dpr_4, dpr_5)


@njit_serial()
def calculate_with_rk4(
        r_pp, pr_pp, pz_pp, gamma_pp, q_pp,
        r_max_plasma, dr_p, parabolic_coefficient,
        a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
        xi_fld, r_fld, dxi, dr, n_xi, n_r,
        max_gamma, p_shape,
        dr_1, dr_2, dr_3, dr_4, dpr_1, dpr_2, dpr_3, dpr_4,
        a2_1, nabla_a2_1, b_t_0_1, b_t_1, psi_1, dr_psi_1, dxi_psi_1,
        a2_2, nabla_a2_2, b_t_0_2, b_t_2, psi_2, dr_psi_2, dxi_psi_2,
        a2_3, nabla_a2_3, b_t_0_3, b_t_3, psi_3, dr_psi_3, dxi_psi_3,
        a2_4, nabla_a2_4, b_t_0_4, b_t_4, psi_4, dr_psi_4, dxi_psi_4):
    """Calculate plasma evolution using the Adams-Bashforth pusher.

    Parameters
    ----------
    r_pp, pr_pp, pz_pp, gamma_pp, q_pp : ndarray
        Radial position, radial momentum, longitudinal momentum,
        Lorentz factor and charge of the plasma particles.
    r_max_plasma : float
        Maximum radial extent of the plasma
    dr_p : float
        Initial radial spacing between plasma particles.
    parabolic_coefficient : float
        Coefficient for the parabolic radial plasma profile.
    a2, nabla_a2, b_t_0 : ndarray
        Laser and beam source fields.
    psi, b_t_bar, rho, chi : ndarray
        Arrays to be filled in during plasma evolution.
    xi_fld, r_fld : ndarray
        Grid coordinates
    dxi, dr : float
        Grid spacing
    n_xi, n_r : int
        Number of grid elements.
    max_gamma : float
        Maximum gamma of the plasma particles.
    p_shape : str
        Particle shape.
    dr_1, ..., dr_4 : ndarray
        Arrays containing the derivative of the radial position of the
        particles at the current slice and the 3 intermediate steps.
    dpr_1, ..., dpr_4 : ndarray
        Arrays containing the derivative of the radial momentum of the
        particles at the current slice and the 3 intermediate steps.
    a2_i, ..., dxi_psi_i : ndarray
        Arrays where the field values at the particle positions at substep i
        will be stored.
    """
    # Loop from the right to the left of the domain.
    for step in range(n_xi):
        slice_i = n_xi - step - 1
        xi = xi_fld[slice_i]

        # Calculate fields at the position of the particles and
        # calculate/deposit psi, b_t_bar, rho and chi at the current slice
        # of the grid.
        calculate_and_deposit_plasma_column(
            slice_i, xi, r_pp, pr_pp, pz_pp, gamma_pp, q_pp,
            r_max_plasma, dr_p, parabolic_coefficient,
            a2_1, nabla_a2_1, b_t_0_1, b_t_1,
            psi_1, dr_psi_1, dxi_psi_1,
            a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
            xi_fld, r_fld, dxi, dr, n_xi, n_r,
            max_gamma, p_shape)

        if slice_i > 0:
            # Evolve plasma to next xi step.
            evolve_plasma_rk4(
                dxi, dr, xi, r_pp, pr_pp, gamma_pp, q_pp,
                r_max_plasma, dr_p, parabolic_coefficient,
                a2, nabla_a2, b_t_0, r_fld, xi_fld,
                dr_1, dr_2, dr_3, dr_4, dpr_1, dpr_2, dpr_3, dpr_4,
                a2_1, nabla_a2_1, b_t_0_1, b_t_1, psi_1, dr_psi_1, dxi_psi_1,
                a2_2, nabla_a2_2, b_t_0_2, b_t_2, psi_2, dr_psi_2, dxi_psi_2,
                a2_3, nabla_a2_3, b_t_0_3, b_t_3, psi_3, dr_psi_3, dxi_psi_3,
                a2_4, nabla_a2_4, b_t_0_4, b_t_4, psi_4, dr_psi_4, dxi_psi_4)


@njit_serial()
def calculate_and_deposit_plasma_column(
        i, xi, r_pp, pr_pp, pz_pp, gamma_pp, q_pp,
        r_max_plasma, dr_p, parabolic_coefficient,
        a2_pp, nabla_a2_pp, b_t_0_pp, b_t_pp, psi_pp, dr_psi_pp, dxi_psi_pp,
        a2, nabla_a2, b_t_0, psi, b_t_bar, rho, chi,
        xi_fld, r_fld, dxi, dr, n_xi, n_r,
        max_gamma, p_shape):
    """Calculate the fields at the current position of the plasma particles
    and calculate/deposit the azimuthal magnetic field from the plasma
    (b_t_bar), the wakefield potential (psi), the plasma charge density (rho)
    and the plasma susceptibility (chi) at the current slice of the grid.

    Parameters
    ----------
    i : int
        Index of the current step.
    xi : float
        Current longitudinal position of the plasma slice.
    r_pp, pr_pp, pz_pp, gamma_pp, q_pp : ndarray
        Radial position, radial momentum, longitudinal momentum,
        Lorentz factor and charge of the plasma particles.
    r_max_plasma : float
        Maximum radial extent of the plasma.
    dr_p : float
        Initial radial spacing between plasma particles.
    parabolic_coefficient : float
        Coefficient for the parabolic radial plasma profile.
    a2_pp, ..., dxi_psi_pp : ndarray
        Arrays where the value of the fields at the particle positions will
        be stored.
    a2, nabla_a2, b_t_0 : ndarray
        Laser and beam source fields.
    psi, b_t_bar, rho, chi : ndarray
        Arrays to be filled in during plasma evolution.
    xi_fld, r_fld : ndarray
        Grid coordinates.
    dxi, dr : float
        Grid spacing.
    n_xi, n_r : int
        Number of grid elements.
    max_gamma : float
        Maximum gamma of the plasma particles.
    p_shape : str
        Particle shape.
    """
    # Gather source terms at position of plasma particles.
    gather_sources_qs_baxevanis(
        a2, nabla_a2, b_t_0, xi_fld[0], xi_fld[-1],
        r_fld[0], r_fld[-1], dxi, dr, r_pp, xi, a2_pp, nabla_a2_pp,
        b_t_0_pp)

    # Get sorted particle indices
    idx = np.argsort(r_pp)

    # Calculate wakefield potential and derivatives at plasma particles.
    calculate_psi_and_derivatives_at_particles(
        r_pp, pr_pp, q_pp, idx, r_max_plasma, dr_p,
        parabolic_coefficient, psi_pp, dr_psi_pp, dxi_psi_pp)

    # Update gamma and pz of plasma particles
    update_gamma_and_pz(gamma_pp, pz_pp, pr_pp, a2_pp, psi_pp)

    # Calculate azimuthal magnetic field from the plasma at the location of
    # the plasma particles.
    calculate_b_theta_at_particles(
        r_pp, pr_pp, q_pp, gamma_pp, psi_pp, dr_psi_pp, dxi_psi_pp,
        b_t_0_pp, nabla_a2_pp, idx, dr_p, b_t_pp)

    # If particles violate the quasistatic condition, slow them down again.
    # This preserves the charge and shows better behavior than directly
    # removing them.
    idx_keep = np.where(gamma_pp >= max_gamma)
    if idx_keep[0].size > 0:
        pz_pp[idx_keep] = 0.
        gamma_pp[idx_keep] = 1.
        pr_pp[idx_keep] = 0.

    # Calculate fields at specified radii for current plasma column.
    calculate_psi(
        r_fld, r_pp, q_pp, idx, r_max_plasma, parabolic_coefficient, psi, i)
    calculate_b_theta(
        r_fld, r_pp, pr_pp, q_pp, gamma_pp, psi_pp, dr_psi_pp, dxi_psi_pp,
        b_t_0_pp, nabla_a2_pp, idx, b_t_bar, i)

    # Deposit rho and chi of plasma column
    w_rho = q_pp / (dr * r_pp * (1 - pz_pp/gamma_pp))
    w_chi = w_rho / gamma_pp
    deposit_plasma_particles(i, r_pp, w_rho, xi_fld[0], r_fld[0], n_xi, n_r,
                             dxi, dr, rho, p_shape=p_shape)
    deposit_plasma_particles(i, r_pp, w_chi, xi_fld[0], r_fld[0], n_xi, n_r,
                             dxi, dr, chi, p_shape=p_shape)


@njit_serial
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


def calculate_beam_source(
        bunch, n_p, n_r, n_xi, r_min, xi_min, dr, dxi, p_shape, b_t):
    """
    Return a (nz+4, nr+4) array with the azimuthal magnetic field
    from a particle distribution. This is Eq. (18) in the original paper.

    """
    # Plasma skin depth.
    s_d = ge.plasma_skin_depth(n_p / 1e6)

    # Get and normalize particle coordinate arrays.
    xi_n = bunch.xi / s_d
    x_n = bunch.x / s_d
    y_n = bunch.y / s_d

    # Calculate particle weights.
    w = - bunch.q / ct.e / (2 * np.pi * dr * dxi * s_d ** 3 * n_p)

    # Obtain charge distribution (using cubic particle shape by default).
    q_dist = np.zeros((n_xi + 4, n_r + 4))
    deposit_3d_distribution(xi_n, x_n, y_n, w, xi_min, r_min, n_xi, n_r, dxi,
                            dr, q_dist, p_shape=p_shape, use_ruyten=True)

    # Remove guard cells.
    q_dist = q_dist[2:-2, 2:-2]

    # Radial position of grid points.
    r_grid_g = (0.5 + np.arange(n_r)) * dr

    # At each grid cell, calculate integral only until cell center by
    # assuming that half the charge is evenly distributed within the cell
    # (i.e., subtract half the charge)
    subs = q_dist / 2

    # At the first grid point along r, subtract an additional 1/4 of the
    # charge. This comes from assuming that the density has to be zero on axis.
    subs[:, 0] += q_dist[:, 0]/4

    # Calculate field by integration.
    b_t[2:-2, 2:-2] += (
        (np.cumsum(q_dist, axis=1) - subs) * dr / np.abs(r_grid_g))

    return b_t