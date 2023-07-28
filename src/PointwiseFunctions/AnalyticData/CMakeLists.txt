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
add_subdirectory(ForceFree)
add_subdirectory(GeneralRelativity)
add_subdirectory(GhGrMhd)
add_subdirectory(GhScalarTensor)
add_subdirectory(GrMhd)
add_subdirectory(NewtonianEuler)
add_subdirectory(Punctures)
add_subdirectory(ScalarTensor)
add_subdirectory(Xcts)
