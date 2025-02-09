import numpy as np
import scipy.constants as ct

from wake_t.fields.analytical_field import AnalyticalField


class CustomBlowoutWakefield(AnalyticalField):
    def __init__(self, n_p, laser, lon_field=None, lon_field_slope=None,
                 foc_strength=None, xi_fields=0.):
        """
        [n_p] = m^-3
        """
        super().__init__()
        self.density = n_p
        self.xi_fields = xi_fields
        self.laser = laser
        self.k = foc_strength
        self.e_z_0 = lon_field
        self.e_z_p = lon_field_slope

        def e_x(x, y, xi, t, ex, constants):
            k = constants[0]
            for i in range(x.shape[0]):
                ex[i] = ct.c * k * x[i]

        def e_y(x, y, xi, t, ey, constants):
            k = constants[0]
            for i in range(x.shape[0]):
                ey[i] = ct.c * k * y[i]

        def e_z(x, y, xi, t, ez, constants):
            e_z_0 = constants[1]
            e_z_p = constants[2]
            xi_fields = constants[3]
            b_w = constants[4]

            xi_off = - xi_fields + (1 - b_w) * ct.c * t
            for i in range(x.shape[0]):
                ez[i] = e_z_0 + e_z_p * (xi[i] + xi_off)

        super().__init__(e_x=e_x, e_y=e_y, e_z=e_z)

    def _pre_gather(self, x, y, xi, t):
        n_p = self.density(t*ct.c)
        b_w = self.laser.get_group_velocity(n_p)
        self.constants = np.array(
            [self.k, self.e_z_0, self.e_z_p, self.xi_fields, b_w])
