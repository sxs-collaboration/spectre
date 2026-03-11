# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  CompletionCriteria.hpp
  CompletionSingleton.hpp
  )

add_subdirectory(Callbacks)
add_subdirectory(Events)
add_subdirectory(PhaseControl)
