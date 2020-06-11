# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY MathFunctions)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Constant.cpp
  Gaussian.cpp
  PowX.cpp
  Sinusoid.cpp
  Sum.cpp
  TensorProduct.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Constant.hpp
  Gaussian.hpp
  MathFunction.hpp
  PowX.hpp
  Sinusoid.hpp
  Sum.hpp
  TensorProduct.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  ErrorHandling
  Options
  )
