# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hll.cpp
  Hllc.cpp
  RegisterDerived.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Factory.hpp
  Hll.hpp
  Hllc.hpp
  RegisterDerived.hpp
  Rusanov.hpp
  )
