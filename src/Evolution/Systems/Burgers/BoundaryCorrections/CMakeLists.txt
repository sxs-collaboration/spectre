# Distributed under the MIT License.
# See LICENSE.txt for details.


spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RegisterDerived.cpp
  Hll.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  BoundaryCorrection.hpp
  Factory.hpp
  Hll.hpp
  RegisterDerived.hpp
  Rusanov.hpp
  )
