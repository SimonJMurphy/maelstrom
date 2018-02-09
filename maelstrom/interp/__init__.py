# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["interp"]

import tensorflow as tf
from ..tf_utils import load_op_library

mod = load_op_library(__file__, "interp_op")
interp = mod.interp


@tf.RegisterGradient("Interp")
def _interp_grad(op, *grads):
    t, x, y = op.inputs
    bf = grads[0]
    bt, by = mod.interp_grad(t, x, y, bf)
    return [bt, None, by]


# @tf.RegisterGradient("InterpGrad")
# def _interp_grad_grad(op, *grads):
#     t, x, bf = op.inputs
#     bbt, bby = grads
#     bbf = mod.interp(t, x, bby)
#     return [None, None, None, bbf]
