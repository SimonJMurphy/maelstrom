# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np
import tensorflow as tf

from .kepler import kepler


class Maelstrom(object):
    """The real deal

    Args:
        time: The array of timestamps.
        mag: The magnitudes measured at ``time``.
        nu: An array of frequencies (in units of ...).
        log_sigma2 (Optional): The log variance of the measurement noise.
        with_eccen (Optional): Should eccentricity be included?
            (default: ``True``)

    """
    T = tf.float64

    def __init__(self, time, mag, nu, log_sigma2=None, with_eccen=True):
        self.time_data = np.atleast_1d(time)
        self.mag_data = np.atleast_1d(mag)
        self.nu_data = np.atleast_1d(nu)
        self.with_eccen = with_eccen

        self.time = tf.constant(self.time_data, dtype=self.T)
        self.mag = tf.constant(self.mag_data, dtype=self.T)

        self._session = None

        # Parameters
        if log_sigma2 is None:
            log_sigma2 = 0.0
        self.log_sigma2 = tf.Variable(log_sigma2, dtype=self.T)
        self.nu = tf.Variable(self.nu_data, dtype=self.T)
        self.period = tf.Variable(1.0, dtype=self.T)
        self.lighttime = tf.Variable(np.zeros_like(self.nu_data), dtype=self.T)
        self.lighttime_inds = tf.Variable(
            np.arange(len(self.nu_data)).astype(np.int32), dtype=tf.int32)
        self.tref = tf.Variable(0.0, dtype=self.T)
        if self.with_eccen:
            self.eccen_param = tf.Variable(-5.0, dtype=self.T)
            self.varpi = tf.Variable(0.0, dtype=self.T)
            self.eccen = 1.0 / (1.0 + tf.exp(-self.eccen_param))

        # Which parameters do we fit for?
        self.params = [
            self.log_sigma2, self.period, self.lighttime, self.tref
        ]
        if self.with_eccen:
            self.params += [self.eccen_param, self.varpi]

        # Set up the model
        self.mean_anom = 2.0*np.pi*(self.time-self.tref)/self.period
        if self.with_eccen:
            self.ecc_anom = kepler(self.mean_anom, self.eccen)
            self.true_anom = 2.0*tf.atan2(
                tf.sqrt(1.0+self.eccen) * tf.tan(0.5*self.ecc_anom),
                tf.sqrt(1.0-self.eccen) + tf.zeros_like(self.time))
            factor = 1.0 - tf.square(self.eccen)
            factor /= 1.0 + self.eccen*tf.cos(self.true_anom)
            self.psi = -factor * tf.sin(self.true_anom + self.varpi)
        else:
            self.psi = -tf.sin(self.mean_anom)

        # Build the design matrix
        ad = tf.gather(self.lighttime, self.lighttime_inds)
        self.tau = ad[None, :] * self.psi[:, None]
        arg = 2.0*np.pi*self.nu[None, :] * (self.time[:, None] - self.tau)
        D = tf.concat([tf.cos(arg), tf.sin(arg),
                       tf.ones((len(self.time_data), 1), dtype=self.T)],
                      axis=1)

        # Solve for the amplitudes and phases of the oscillations
        DTD = tf.matmul(D, D, transpose_a=True)
        DTy = tf.matmul(D, self.mag[:, None], transpose_a=True)
        W_hat = tf.linalg.solve(DTD, DTy)

        # Finally, the model and the chi^2 objective:
        self.model = tf.squeeze(tf.matmul(D, W_hat))
        self.chi2 = tf.reduce_sum(tf.square(self.mag - self.model))
        self.chi2 *= tf.exp(-self.log_sigma2)
        self.chi2 += len(self.time_data) * self.log_sigma2

        # Initialize all the variables
        self.run(tf.global_variables_initializer())

    def run(self, *args, **kwargs):
        return self.get_session().run(*args, **kwargs)

    def __del__(self):
        if self._session is not None:
            self._session.close()

    def get_session(self, *args, **kwargs):
        if self._session is None:
            self._session = tf.Session(*args, **kwargs)
        return self._session

    def init_from_orbit(self, period, lighttime, tref=0.0, eccen=1e-5,
                        varpi=0.0):
        """Initialize the parameters based on an orbit estimate

        Args:
            period: The orbital period in units of ``time``.
            lighttime: The projected light travel time in units of ``time``
                (:math:`a_1\,\sin(i)/c`).
            tref: The reference time in units of ``time``.
            eccen: The orbital eccentricity.
            varpi: The angle of the ascending node in radians.

        """
        ops = []
        ops.append(tf.assign(self.period, period))
        ops.append(tf.assign(self.lighttime,
                             lighttime + tf.zeros_like(self.lighttime)))
        ops.append(tf.assign(self.tref, tref))
        if self.with_eccen:
            ops.append(tf.assign(self.eccen_param,
                                 np.log(eccen) - np.log(1.0 - eccen)))
            ops.append(tf.assign(self.varpi, varpi))
        self.run(ops)

    def get_fit_parameters(self):
        """Get the list of parameters that are being fit"""
        return self.params

    def set_fit_parameters(self, params):
        """Set the list of parameters that should be fit

        Args:
            params: A list of TensorFlow variables that should be fit.

        """
        self.params = params

    def optimize(self, params=None, **kwargs):
        if params is None:
            params = self.params
        kwargs["method"] = kwargs.get("method", "L-BFGS-B")
        opt = tf.contrib.opt.ScipyOptimizerInterface(self.chi2, params,
                                                     **kwargs)
        return opt.minimize(self.get_session())

    def get_lighttime_estimates(self):
        sigma = 1.0 / tf.sqrt(-tf.diag_part(tf.hessians(-0.5*self.chi2,
                                                        self.lighttime)[0]))
        return self.run([self.lighttime, sigma])

    def pin_lighttime_values(self, double=False):
        if double:
            pass
        else:
            pass
