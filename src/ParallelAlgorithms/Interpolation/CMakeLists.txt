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
  Interpolate.hpp
  InterpolatedVars.hpp
  InterpolationTarget.hpp
  InterpolationTargetDetail.hpp
  Interpolator.hpp
  Intrp.hpp
  IntrpOptionHolders.hpp
  PointInfoTag.hpp
  Tags.hpp
  TagsMetafunctions.hpp
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )

# Some of these dependencies should be removed, see issue
# https://github.com/sxs-collaboration/spectre/issues/3561
target_link_libraries(
  ${LIBRARY}
  PUBLIC
  ApparentHorizons
  DataStructures
  Domain
  ErrorHandling
  GeneralRelativitySolutions
  Interpolation
  Logging
  Options
  Parallel
  Serialization
  Spectral
  SphericalHarmonics
  Time
  Utilities
  INTERFACE
  DgSubcell
  EventsAndTriggers
  GeneralizedHarmonic
  Observer
  )

add_subdirectory(Actions)
add_subdirectory(Callbacks)
add_subdirectory(Events)
add_subdirectory(Protocols)
add_subdirectory(Targets)
