# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Constant.cpp
  GaussianPlusConstant.cpp
  RegisterDerivedWithCharm.cpp
  TimeDependentTripleGaussian.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Constant.hpp
  DampingFunction.hpp
  GaussianPlusConstant.hpp
  RegisterDerivedWithCharm.hpp
  Tags.hpp
  TimeDependentTripleGaussian.hpp
  )
