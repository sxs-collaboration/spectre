# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY GhConstraintDamping)

add_spectre_library(${LIBRARY})

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

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  Utilities
  PRIVATE
  FunctionsOfTime
  )
