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

# Create a library that can be linked into an external program. Notes:
# - This is still experimental. The bundled library works (see
#   Test_BundledExporter.cpp), but building the library on one system and then
#   distributing it to another (i.e. making the library relocatable) will likely
#   require some more work. Linking in external dependencies statically would
#   probably help, so they don't have to be installed on the target system (e.g.
#   HDF5, yaml-cpp, etc).
# - A shared library seems to work better than a static library, since it
#   includes symbols from linked static libs.
# - The `TARGET_OBJECTS` generator expression links in the objects from the
#   Exporter library, because otherwise they wouldn't be used and therefore
#   wouldn't be linked in.
# - The `EXPORTER_LINKED_LIBS` are needed to resolve the symbols for the
#   `TARGET_OBJECTS`, just like in the Exporter library (of which this library
#   is basically an extended copy).
# - The `BundledExporter.cpp` file and the `CharmModuleInit` object library
#   define some symbols for compatility with Charm++ and InfoAtLink.
set(BUNDLED_EXPORTER_LIB BundledExporter)
add_library(${BUNDLED_EXPORTER_LIB} SHARED
  BundledExporter.cpp)
get_target_property(EXPORTER_LINKED_LIBS ${LIBRARY} LINK_LIBRARIES)
target_link_libraries(
  ${BUNDLED_EXPORTER_LIB}
  PRIVATE
  $<TARGET_OBJECTS:${LIBRARY}>
  ${EXPORTER_LINKED_LIBS}
  Informer
  CharmModuleInit
  Charmxx::charmxx
  )
add_dependencies(${BUNDLED_EXPORTER_LIB} ${LIBRARY})
# Also export the public header so that external programs can include it
configure_file(
  Exporter.hpp
  ${CMAKE_BINARY_DIR}/include/spectre/Exporter.hpp
  COPYONLY)
