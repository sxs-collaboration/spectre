# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_M1Grey")

set(LIBRARY_SOURCES
  Test_M1Closure.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/RadiationTransport/M1Grey/"
  "${LIBRARY_SOURCES}"
  "M1Grey"
  )
