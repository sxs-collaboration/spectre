# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Ccz4)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ATilde.cpp
  Christoffel.cpp
  DerivChristoffel.cpp
  DerivLapse.cpp
  DerivZ4Constraint.cpp
  Ricci1.cpp
  Ricci2.cpp
  Ricci3.cpp
  RicciScalarPlusDivergenceZ4Constraint.cpp
  TimeDerivative.cpp
  Z4Constraint.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ATilde.hpp
  Christoffel.hpp
  DerivChristoffel.hpp
  DerivLapse.hpp
  DerivZ4Constraint.hpp
  Ricci.hpp
  Ricci.tpp
  RicciScalarPlusDivergenceZ4Constraint.hpp
  System.hpp
  Tags.hpp
  TagsDeclarations.hpp
  TempTags.hpp
  TimeDerivative.hpp
  Z4Constraint.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  DataStructures
  GeneralRelativity
  Options
  Utilities
  INTERFACE
  )
