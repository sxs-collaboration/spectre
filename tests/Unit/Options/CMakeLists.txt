# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Options")

set(LIBRARY_SOURCES
  Test_CustomTypeConstruction.cpp
  Test_Factory.cpp
  Test_Options.cpp
  )

add_test_library(
  ${LIBRARY}
  "Options/"
  "${LIBRARY_SOURCES}"
  "Utilities"
  )
