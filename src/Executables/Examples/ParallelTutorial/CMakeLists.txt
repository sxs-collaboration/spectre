# Distributed under the MIT License.
# See LICENSE.txt for details.

# [add_spectre_executable]
set(EXECUTABLE MinimalExample)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  MinimalExecutable.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Charmxx::main
  Informer
  Options
  Parallel
  Utilities
  )
# [add_spectre_executable]
