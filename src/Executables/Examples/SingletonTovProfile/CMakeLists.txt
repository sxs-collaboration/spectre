# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  DataStructures
  GeneralRelativitySolutions
  Hydro
  Informer
  IO
  MathFunctions
  Utilities
  )

add_spectre_parallel_executable(
  SingletonTovProfile
  SingletonTovProfile
  Executables/Examples/SingletonTovProfile
  Metavars
  "${LIBS_TO_LINK}"
  )
