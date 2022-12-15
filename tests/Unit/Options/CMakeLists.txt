# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Options")

set(LIBRARY_SOURCES
  Test_Auto.cpp
  Test_Comparator.cpp
  Test_CustomTypeConstruction.cpp
  Test_Factory.cpp
  Test_FactoryHelpers.cpp
  Test_Options.cpp
  Test_StdComplex.cpp
  )

add_test_library(
  ${LIBRARY}
  "Options/"
  "${LIBRARY_SOURCES}"
  "Options;Utilities"
  )
