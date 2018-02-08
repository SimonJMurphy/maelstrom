# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["load_op_library"]

import os
import sysconfig
import tensorflow as tf


def load_op_library(module_file, name):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    dirname = os.path.dirname(os.path.abspath(module_file))
    return tf.load_op_library(os.path.join(dirname, name + suffix))
