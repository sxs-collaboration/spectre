# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Cce")

set(LIBRARY_SOURCES
  Test_Equations.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Cce/"
  "${LIBRARY_SOURCES}"
  "Cce"
  )
