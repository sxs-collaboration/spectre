# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Exporter)

# Use 'add_library' rather than 'add_spectre_library' for multiple reasons:
# - Don't build with PCH, because otherwise we'd have to depend the PCH on
#   OpenMP (would probably be fine, but then _everything_ would link with OpenMP
#   and it's probably better to avoid that).
# - Don't try to stub the object files because the BundledExporter library
#   (defined below) uses them.
add_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Exporter.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Exporter.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Domain
  DomainCreators
  FunctionsOfTime
  H5
  Interpolation
  Serialization
  Utilities
  )

# Link OpenMP if available
if(TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(${LIBRARY} PRIVATE OpenMP::OpenMP_CXX)
endif()

add_subdirectory(Python)
