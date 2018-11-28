# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY VariableFixing)

set(LIBRARY_SOURCES
  FixToAtmosphere.cpp
  LimitLorentzFactor.cpp
  RadiallyFallingFloor.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  )
