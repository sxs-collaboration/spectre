# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  ExecutableStatus
  MODULE_PATH tools/Status
  PYTHON_FILES
  __init__.py
  EvolveGhBinaryBlackHole.py
  EvolveGhSingleBlackHole.py
  ExecutableStatus.py
  SolveXcts.py
)
