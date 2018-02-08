# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["InterpMaelstrom"]

import numpy as np
import tensorflow as tf

from .interp import interp
from .maelstrom import Maelstrom


class InterpMaelstrom(Maelstrom):

    def setup_orbit_model(self, interp_x=None, interp_y=None):
        if interp_x is None:
            interp_x = np.linspace(self.time_data.min(), self.time_data.max(),
                                   100)
        interp_x = np.atleast_1d(interp_x)
        inds = np.argsort(interp_x)
        interp_x = interp_x[inds]
        self.interp_x = tf.constant(interp_x, dtype=self.T, name="interp_x")

        if interp_y is None:
            interp_y = np.zeros_like(interp_x)
        interp_y = np.atleast_1d(interp_y)[inds]
        self.interp_y = tf.Variable(interp_y, dtype=self.T, name="interp_y")

        self.params = [self.interp_y]

        self.psi = interp(self.time, self.interp_x, self.interp_y)
        self.tau = self.psi[:, None] + tf.zeros((len(self.time_data),
                                                 len(self.nu_data)),
                                                dtype=self.T)
