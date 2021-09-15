# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Ccz4")

set(LIBRARY_SOURCES
  Test_Christoffel.cpp
  Test_DerivLapse.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/Ccz4/"
  "${LIBRARY_SOURCES}"
  "Ccz4;DataBoxTestHelpers;DataStructures;Utilities"
  )
