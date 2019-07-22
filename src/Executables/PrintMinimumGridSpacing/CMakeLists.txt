# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE PrintMinimumGridSpacing)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ${EXECUTABLE}.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  ${BLAS_LIBRARIES}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  DomainCreators
  Informer
  Utilities
  ${YAMLCPP_LIBRARIES}
  )

set_target_properties(
  ${EXECUTABLE}
  PROPERTIES LINK_FLAGS "-nomain-module -nomain"
  )
