# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Test_PuncturesPointwiseFunctions)

set(LIBRARY_SOURCES
  Test_AdmIntegrals.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/Punctures/"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Spectral
  PuncturesPointwiseFunctions
  Utilities
  )

add_subdirectory(Python)
