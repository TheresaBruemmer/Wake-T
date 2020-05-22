
import matplotlib.pyplot as plt
from aptools.plotting.quick_diagnostics import slice_analysis

from wake_t.beamline_elements import PlasmaStage
from wake_t.driver_witness import LaserPulse
from wake_t.utilities.bunch_generation import get_matched_bunch
from wake_t.data_analysis import analyze_bunch_list

# create laser driver
laser = LaserPulse(100e-6, l_0=800e-9, w_0=70e-6, a_0=0.8, tau=30e-15)

# create bunch (matched to a focusing srength of 1.5e5 T)
en = 0.3e-6  # m
ene = 200  # units of beta*gamma
ene_sp = 0.3  # %
xi_c = laser.xi_c - 55e-6  # m
s_t = 1  # fs
q_tot = 1  # pC
n_part = 1e4
bunch = get_matched_bunch(en, en, ene, ene_sp, s_t, xi_c, q_tot, n_part,
                          k_x=130000)

# create plasma stage
plasma = PlasmaStage(
    1e-2, 1e23, driver=laser, wakefield_model='cold_fluid_1d', n_out=20,
    laser_evolution=True, laser_z_foc=0, r_max=70e-6,  xi_min=40e-6,
    xi_max=120e-6, n_r=70, n_xi=50)

# start tracking
bunch_list = plasma.track(bunch, out_initial=True)

# analyze bunch evolution
params_evolution = analyze_bunch_list(bunch_list)

# quick plot of results
z = params_evolution['prop_dist'] * 1e2
fig_1 = plt.figure()
plt.subplot(411)
plt.plot(z, params_evolution['beta_x']*1e3)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\beta_x$ [mm]")
plt.subplot(412)
plt.plot(z, params_evolution['emitt_x']*1e6)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\epsilon_{nx}$ [$\\mu$m]")
plt.subplot(413)
plt.plot(z, params_evolution['rel_ene_spread']*100)
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.ylabel("$\\frac{\\Delta \\gamma}{\\gamma}$ [%]")
plt.subplot(414)
plt.plot(z, params_evolution['avg_ene'])
plt.xlabel("z [mm]")
plt.ylabel("$\\gamma$")
plt.tight_layout()

fig_2 = plt.figure()
slice_analysis(bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz,
               bunch.q, fig=fig_2)
plt.show()
