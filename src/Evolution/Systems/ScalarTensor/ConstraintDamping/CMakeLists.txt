# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  GaussianPlusConstant.cpp
  RegisterDerivedWithCharm.cpp
)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstraintGammas.hpp
  DampingFunction.hpp
  GaussianPlusConstant.hpp
  RegisterDerivedWithCharm.hpp
  Tags.hpp
)
