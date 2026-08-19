# Distributed under the MIT License.
# See LICENSE.txt for details.

"""The bin directory that a simulation runs from

A scheduled job runs unsupervised long after it was submitted, so it must not
depend on the build directory it was scheduled from: that gets recompiled,
switched to another branch, and eventually deleted. Everything the job needs
after submission is therefore copied into a 'bin' directory of the simulation,
and the job runs from there.

This module holds the layout of that directory, and the code that creates one
and finds it again.
"""

# Layout of the directory that holds the CLI wrapper and the Python package.
# 'cmake/SpectreSetupPythonPackage.cmake' configures the wrapper with the same
# names.
BIN_DIR_NAME = "bin"
PYTHON_DIR_NAME = "python"
