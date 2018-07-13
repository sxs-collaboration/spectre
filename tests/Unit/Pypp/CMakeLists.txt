# Distributed under the MIT License.
# See LICENSE.txt for details.

SET(LIBRARY_SOURCES
    SetupLocalPythonEnvironment.cpp
    Test_Pypp.cpp
    )

add_test_library(
  "Test_Pypp"
  "Pypp"
  "${LIBRARY_SOURCES}"
  "ErrorHandling"
  )
