# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ParallelInterpolation)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  InterpolationTargetDetail.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InterpolatedVars.hpp
  InterpolationTarget.hpp
  InterpolationTargetDetail.hpp
  Interpolator.hpp
  Intrp.hpp
  IntrpOptionHolders.hpp
  PointInfoTag.hpp
  Tags.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  Blas
  Boost::boost
  DataStructures
  DgSubcell
  Domain
  DomainStructure
  ErrorHandling
  EventsAndTriggers
  GSL::gsl
  Interpolation
  Logging
  Options
  Spectral
  INTERFACE
  SystemUtilities
  )

add_subdirectory(Actions)
add_subdirectory(Callbacks)
add_subdirectory(Events)
add_subdirectory(Targets)
