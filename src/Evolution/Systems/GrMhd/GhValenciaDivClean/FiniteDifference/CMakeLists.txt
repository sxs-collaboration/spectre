# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Derivatives.cpp
  FilterOptions.cpp
  MonotonisedCentral.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Derivatives.hpp
  Factory.hpp
  FilterOptions.hpp
  FiniteDifference.hpp
  MonotonisedCentral.hpp
  Reconstructor.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  RegisterDerivedWithCharm.hpp
  Tag.hpp
  )
