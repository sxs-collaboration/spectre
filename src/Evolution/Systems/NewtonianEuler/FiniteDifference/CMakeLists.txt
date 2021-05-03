# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  MonotisedCentral.cpp
  Reconstructor.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  FiniteDifference.hpp
  MonotisedCentral.hpp
  ReconstructWork.hpp
  ReconstructWork.tpp
  Reconstructor.hpp
  RegisterDerivedWithCharm.hpp
  )
