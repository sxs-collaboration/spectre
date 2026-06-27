# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Strahlkorper")

set(LIBRARY_SOURCES
  Test_ChangeCenterOfStrahlkorper.cpp
  Test_Strahlkorper.cpp
  Test_StrahlkorperDataBox.cpp
  Test_StrahlkorperFunctions.cpp
  Test_Tags.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  GrSurfaces
  H5
  RootFinding
  SphericalHarmonics
  SphericalHarmonicsHelpers
  Strahlkorper
  StrahlkorperIO
  Utilities
  )

add_subdirectory(IO)
add_subdirectory(Python)
