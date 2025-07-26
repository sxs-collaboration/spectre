# Distributed under the MIT License.
# See LICENSE.txt for details.

if (NOT TARGET Kokkos::kokkos)
  return()
endif()

# Add a library with device code to test linking it into an executable
add_spectre_library(
  TestKernel
  TestKernel.cpp
)
target_link_libraries(TestKernel PUBLIC Kokkos::kokkos)

# Add a test executable that links against the TestKernel library
set(LIBRARY "Test_Kokkos")

set(LIBRARY_SOURCES
  Test_Kokkos.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  TestKernel
  SpectreKokkos
  )
