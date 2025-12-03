# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AtomicInboxBoundaryData.hpp
  BackgroundGrVars.hpp
  BoundaryData.hpp
  CleanMortarHistory.hpp
  CleanMortarHistory.tpp
  DgElementArray.hpp
  InboxBoundaryData.hpp
  InboxTags.hpp
  InterfaceDataPolicy.hpp
  InterpolatedBoundaryData.hpp
  MortarData.hpp
  MortarDataHolder.hpp
  MortarInfo.hpp
  MortarTags.hpp
  NormalVectorTags.hpp
  TimeDerivativeDecisions.hpp
  UsingSubcell.hpp
  )

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AtomicInboxBoundaryData.cpp
  BoundaryData.cpp
  InboxBoundaryData.cpp
  InboxTags.cpp
  InterfaceDataPolicy.cpp
  InterpolatedBoundaryData.cpp
  MortarData.cpp
  MortarDataHolder.cpp
  MortarInfo.cpp
  )

add_subdirectory(Actions)
add_subdirectory(Initialization)
add_subdirectory(Limiters)
add_subdirectory(Messages)
