# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAlgorithmsEvents")

set(LIBRARY_SOURCES
  Test_ObserveErrorNorms.cpp
  Test_ObserveFields.cpp
  Test_ObserveVolumeIntegrals.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Events/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling;IO;Time;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
