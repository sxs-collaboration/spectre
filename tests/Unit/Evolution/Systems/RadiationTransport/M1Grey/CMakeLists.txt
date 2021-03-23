# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_M1Grey")

set(LIBRARY_SOURCES
  BoundaryCorrections/Test_Rusanov.cpp
  Test_Actions.cpp
  Test_Fluxes.cpp
  Test_M1Closure.cpp
  Test_M1HydroCoupling.cpp
  Test_Sources.cpp
  Test_Tags.cpp
  Test_TimeDerivativeTerms.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/RadiationTransport/M1Grey/"
  "${LIBRARY_SOURCES}"
  "M1Grey"
  "Boost::boost"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
