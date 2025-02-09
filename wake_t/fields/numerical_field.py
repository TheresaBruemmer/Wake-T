"""Contains the base class for all numerical fields."""

import numpy as np

from .base import Field


class NumericalField(Field):
    """Base class for all fields that are computed numerically.

    All numerical fields must have a time step `dt_update` which determines how
    often they will be updated (i.e., recalculated). Every time `update` is
    called, the internal time is advanced by one step and the field is
    recalculated. The class also provides methods for initializing and evolving
    field properties (e.g., a laser pulse envelope).

    Each subclass must implement the logic for calculating and gathering the
    field and, optionally, for initializing and evolving any field properties.
    """

    def __init__(self, dt_update, openpmd_diag_supported=False,
                 force_even_updates=False):
        """Initialize the field.

        Parameters
        ----------
        dt_update : float
            Update period (in seconds) of the field.
        openpmd_diag_supported : bool
            Whether openPMD diagnostics are supported by the field.
        force_even_updates : bool
            During tracking, it can happen that the total simulation time
            is not an integer multiple of `dt_update`, so that the last
            update is used for less time than the others. If set to True,
            this parameter will modify `dt_update` (making it smaller, never
            larger) so that the total tracking time is an integer multiple
            of `dt_update`. This makes sure also that the fields are
            updated one last time exactly at the end of the stage.
        """
        super().__init__(openpmd_diag_supported=openpmd_diag_supported)
        self.dt_update = dt_update
        self.force_even_updates = force_even_updates
        self.initialized = False

    def update(self, bunches):
        """Update field to the next time step (`dt_update`).

        Parameters
        ----------
        bunches : list
            List of `ParticleBunch`es that can be used to recompute/update the
            fields.
        """
        if not self.initialized:
            self.initialize_properties(bunches)
        else:
            self.evolve_properties(bunches)
        self.calculate_field(bunches)

    def initialize_properties(self, bunches):
        self.t = 0.
        self._initialize_properties(bunches)
        self.initialized = True

    def evolve_properties(self, bunches):
        self.t += self.dt_update
        self._evolve_properties(bunches)

    def calculate_field(self, bunches):
        self._calculate_field(bunches)

    def adjust_dt(self, t_final):
        if self.force_even_updates:
            n_updates = np.ceil(t_final / self.dt_update)
            self.dt_update = t_final / n_updates

    def _initialize_properties(self, bunches):
        pass

    def _evolve_properties(self, bunches):
        pass

    def _calculate_field(self, bunches):
        raise NotImplementedError
