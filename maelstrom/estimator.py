# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["estimate_frequencies"]

import numpy as np
import tensorflow as tf
from astropy.stats import LombScargle


def estimate_frequencies(x, y, max_peaks=6, oversample=4.0):
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    freq = np.arange(df, 2 * ny, df / oversample)
    power = LombScargle(x, y).power(freq)

    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0*ny - freq0

        m = np.abs(freq[peak_inds] - alias) > 25*df
        m &= np.abs(freq[peak_inds] - freq0) > 25*df

        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)

    # Optimize the model
    T = tf.float64
    t = tf.constant(x, dtype=T)
    f = tf.constant(y, dtype=T)
    nu = tf.Variable(peaks, dtype=T)
    arg = 2*np.pi*nu[None, :]*t[:, None]
    D = tf.concat([tf.cos(arg), tf.sin(arg),
                   tf.ones((len(x), 1), dtype=T)],
                  axis=1)

    # Solve for the amplitudes and phases of the oscillations
    DTD = tf.matmul(D, D, transpose_a=True)
    DTy = tf.matmul(D, f[:, None], transpose_a=True)
    w = tf.linalg.solve(DTD, DTy)
    model = tf.squeeze(tf.matmul(D, w))
    chi2 = tf.reduce_sum(tf.square(f - model))

    opt = tf.contrib.opt.ScipyOptimizerInterface(chi2, [nu],
                                                 method="L-BFGS-B")
    with tf.Session() as sess:
        sess.run(nu.initializer)
        opt.minimize(sess)
        return sess.run(nu)
