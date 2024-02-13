# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY AmrPolicies)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  EnforcePolicies.cpp
  Isotropy.cpp
  Policies.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  EnforcePolicies.hpp
  Isotropy.hpp
  Policies.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Options
  )
