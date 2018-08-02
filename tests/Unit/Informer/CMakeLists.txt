# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Informer")

set(LIBRARY_SOURCES
  Test_Verbosity.cpp
  )

add_test_library(
  ${LIBRARY}
  "Informer/"
  "${LIBRARY_SOURCES}"
  "Informer;Utilities"
  )
