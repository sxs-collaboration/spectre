# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Serialization)

add_spectre_library(${LIBRARY} INTERFACE)

# spectre_target_sources(
#   ${LIBRARY}
#   PUBLIC
#   )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CharmPupable.hpp
  PupBoost.hpp
  PupStlCpp11.hpp
  PupStlCpp17.hpp
  RegisterDerivedClassesWithCharm.hpp
  Serialize.hpp
  )

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  Charmxx::pup
  Utilities
  )
