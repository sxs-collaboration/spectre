# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_python_add_module(
  Bbh
  MODULE_PATH Pipelines
  PYTHON_FILES
  __init__.py
  ControlId.py
  FindHorizon.py
  InitialData.py
  InitialData.yaml
  Inspiral.py
  Inspiral.yaml
  PostprocessId.py
  Ringdown.py
  Ringdown.yaml
)

# Create a target to compile all executables for this pipeline
set(PIPELINE_TARGET bbh)
add_custom_target(${PIPELINE_TARGET})
add_dependencies(
  ${PIPELINE_TARGET}
  EvolveGhBinaryBlackHole
  EvolveGhSingleBlackHole
  SolveXcts
  )
