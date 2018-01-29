# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["kepler"]

import os
import sysconfig

import tensorflow as tf

suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
mod = tf.load_op_library(os.path.join(dirname, "kepler_op" + suffix))

kepler = mod.kepler


@tf.RegisterGradient("Kepler")
def _kepler_grad(op, *grads):
    M, e = op.inputs
    E = op.outputs[0]
    bE = grads[0]
    bM = bE / (1.0 - e * tf.cos(E))
    be = tf.reduce_sum(tf.sin(E) * bM)

    return [bM, be]

    # args = list(op.inputs) + list(op.outputs) + list(grads)
    # return mod.kepler_grad(*args)
