# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GhGrMhdSolutions")

set(LIBRARY_SOURCES
  Test_InstantiateWrappedGr.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/AnalyticSolutions/GhGrMhd"
  "${LIBRARY_SOURCES}"
  "GeneralRelativitySolutions;GhGrMhdSolutions;GrMhdSolutions;Options;Utilities"
  )
