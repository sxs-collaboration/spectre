# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ScalarTensorPointwise)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  RampUpFunction.cpp
  ScalarCharge.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ConstraintDampingTags.hpp
  ConstraintGammas.hpp
  RampUpFunction.hpp
  ScalarCharge.hpp
  SourceTags.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  INTERFACE
  ErrorHandling
  )
