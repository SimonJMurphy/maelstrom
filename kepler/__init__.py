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
    args = list(op.inputs) + list(op.outputs) + list(grads)
    return mod.kepler_grad(*args)
