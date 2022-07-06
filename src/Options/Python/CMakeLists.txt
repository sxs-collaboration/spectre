# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyOptions")

spectre_python_add_module(
  Options
  PYTHON_EXECUTABLES
  ExtractInputSourceYamlFromH5.py
)
