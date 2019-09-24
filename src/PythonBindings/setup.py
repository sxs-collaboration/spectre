#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from distutils.core import setup

setup(
    name='spectre',
    version='${SpECTRE_VERSION}',
    description="Python bindings for SpECTRE",
    author="SXS collaboration",
    url="https://spectre-code.org",
    packages=['spectre'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
