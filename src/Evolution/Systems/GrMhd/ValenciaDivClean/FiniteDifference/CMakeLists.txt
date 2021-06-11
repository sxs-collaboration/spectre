# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  FiniteDifference.hpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  Tag.hpp
  )
