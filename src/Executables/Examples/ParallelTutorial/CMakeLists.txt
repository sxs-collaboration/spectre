# Distributed under the MIT License.
# See LICENSE.txt for details.

# [add_spectre_executable]
set(LIBS_TO_LINK
  Informer
  Utilities
  )

add_spectre_parallel_executable(
  MinimalExample
  MinimalExecutable
  Executables/Examples/ParallelTutorial
  Metavariables
  "${LIBS_TO_LINK}"
  )
# [add_spectre_executable]
