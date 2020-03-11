# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyPoissonTestHelpers")

spectre_python_add_module(
  Poisson
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "TestHelpers/Elliptic/Systems"
  SOURCES
  Bindings.cpp
  DgSchemes.cpp
  )

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainCreators
  PoissonTestHelpers
  )

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PyDomainCreators
  )
