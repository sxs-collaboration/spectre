# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AoWeno.cpp
  MonotonisedCentral.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AoWeno.hpp
  Factory.hpp
  FiniteDifference.hpp
  MonotonisedCentral.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  Tag.hpp
  )
