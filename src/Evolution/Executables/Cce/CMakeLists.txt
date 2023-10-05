# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Charmxx::main
  Cce
  Time
  Interpolation
  Informer
  Spectral
  LinearOperators
  Observer
  Options
  Parallel
  Serialization
  Utilities
  )

function(add_cce_executable EXECUTABLE BOUNDARY_COMPONENT)
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    CharacteristicExtract.cpp
  )
  target_compile_definitions(
    ${EXECUTABLE}
    PRIVATE
    BOUNDARY_COMPONENT=${BOUNDARY_COMPONENT}
    )
  target_link_libraries(${EXECUTABLE} PRIVATE ${LIBS_TO_LINK})
endfunction()

add_cce_executable(
  CharacteristicExtract
  Cce::H5WorldtubeBoundary
  )
add_cce_executable(
  AnalyticTestCharacteristicExtract
  Cce::AnalyticWorldtubeBoundary
  )
