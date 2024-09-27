# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY FunctionsOfTime)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FixedSpeedCubic.cpp
  IntegratedFunctionOfTime.cpp
  OutputTimeBounds.cpp
  PiecewisePolynomial.cpp
  QuaternionFunctionOfTime.cpp
  QuaternionHelpers.cpp
  RegisterDerivedWithCharm.cpp
  SettleToConstant.cpp
  SettleToConstantQuaternion.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FixedSpeedCubic.hpp
  FunctionOfTime.hpp
  IntegratedFunctionOfTime.hpp
  OptionTags.hpp
  OutputTimeBounds.hpp
  PiecewisePolynomial.hpp
  QuaternionFunctionOfTime.hpp
  QuaternionHelpers.hpp
  RegisterDerivedWithCharm.hpp
  SettleToConstant.hpp
  SettleToConstantQuaternion.hpp
  Tags.hpp
  ThreadsafeList.hpp
  ThreadsafeList.tpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  Options
  Serialization
  Utilities
  PRIVATE
  ErrorHandling
  )
