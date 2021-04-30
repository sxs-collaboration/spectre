# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AnalyticData)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AnalyticData.hpp
  Tags.hpp
  )

add_subdirectory(Burgers)
add_subdirectory(CurvedWaveEquation)
add_subdirectory(Elasticity)
add_subdirectory(GrMhd)
add_subdirectory(NewtonianEuler)
add_subdirectory(ScalarAdvection)
add_subdirectory(Xcts)
