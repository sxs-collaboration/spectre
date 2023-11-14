# Distributed under the MIT License.
# See LICENSE.txt for details.

# If XSIMD exists, set up a wrapper target.
#
# Having a wrapper target will make it easier to replace XSIMD
# with another SIMD library if we ever need to.
set(LIBRARY Simd)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Simd.hpp
  )

if (TARGET xsimd)
  target_link_libraries(
    ${LIBRARY}
    INTERFACE
    xsimd
    )

  message(STATUS "Enabling simd support")
endif()
