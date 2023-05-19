# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY SizeControlErrors)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Info.cpp
  AhSpeed.cpp
  DeltaR.cpp
  Error.cpp
  Initial.cpp
  RegisterDerivedWithCharm.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Info.hpp
  State.hpp
  AhSpeed.hpp
  DeltaR.hpp
  Error.hpp
  Initial.hpp
  RegisterDerivedWithCharm.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  Boost::boost
  DataStructures
  ErrorHandling
  GSL::gsl
  Interpolation
  Parallel
  )
