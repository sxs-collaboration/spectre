# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Initialization")

set(LIBRARY_SOURCES
  Test_MergeIntoDataBox.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling;Utilities"
  )

add_dependencies(
  ${LIBRARY}
  module_ConstGlobalCache
  )
