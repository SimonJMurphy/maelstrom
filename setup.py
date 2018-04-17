#!/usr/bin/env python

import os
import tensorflow as tf
from setuptools import setup, Extension


# include_dirs = [".", tf.sysconfig.get_include()]
# include_dirs.append(os.path.join(
#     include_dirs[1], "external/nsync/public"))

compile_flags = tf.sysconfig.get_compile_flags()
compile_flags += ["-std=c++11", "-stdlib=libc++", "-O2", "-undefined dynamic_lookup"]
link_flags = tf.sysconfig.get_link_flags()

extensions = [
    Extension(
        "maelstrom.kepler.kepler_op",
        sources=["maelstrom/kepler/kepler_op.cc"],
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
    ),
    Extension(
        "maelstrom.interp.interp_op",
        sources=[
            "maelstrom/interp/searchsorted_op.cc",
        ],
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    name="maelstrom",
    license="MIT",
    packages=["maelstrom", "maelstrom.kepler", "maelstrom.interp"],
    url = 'https://github.com/SimonJMurphy/maelstrom',
    install_requires=['numpy>=1.10','astropy>=1.0','tensorflow'],
    ext_modules=extensions,
    zip_safe=True,
)
