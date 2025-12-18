# Distributed under the MIT License.
# See LICENSE.txt for details.

# Having a wrapper target will make it easier to replace autodiff
# with another auto-differentiation library if we ever need to.

set(LIBRARY Autodiff)

add_spectre_library(${LIBRARY} INTERFACE)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Autodiff.hpp
  )

if (SPECTRE_AUTODIFF)
  target_link_libraries(
    ${LIBRARY}
    INTERFACE
    autodiff::autodiff
    Simd
    )

  target_compile_definitions(
    ${LIBRARY}
    INTERFACE
    SPECTRE_AUTODIFF
    )

  message(STATUS "Enabling autodiff support")
endif()
