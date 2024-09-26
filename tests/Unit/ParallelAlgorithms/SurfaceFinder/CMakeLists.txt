# Distributed under the MIT License.
# See LICENSE.txt for details.
set(LIBRARY "Test_SurfaceFinder")

set(LIBRARY_SOURCES
  Test_SurfaceFinder.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  Domain
  DomainCreators
  MathFunctions
  Spectral
  SurfaceFinder
  )

spectre_add_python_bindings_test(
  "SurfaceFinder.FindRadialSurface"
  Test_FindRadialSurface.py
  "Python"
  None
  TIMEOUT 20)
