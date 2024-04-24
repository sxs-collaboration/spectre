# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  Cce
  Charmxx::main
  Informer
  Interpolation
  LinearOperators
  Observer
  Options
  Parallel
  Serialization
  Spectral
  SpinWeightedSphericalHarmonics
  Time
  Utilities
  )

function(add_cce_executable EXECUTABLE BOUNDARY_COMPONENT SYSTEM)
  add_spectre_executable(
    ${EXECUTABLE}
    EXCLUDE_FROM_ALL
    ${SYSTEM}.cpp
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
  CharacteristicExtract
  )
add_cce_executable(
  AnalyticTestCharacteristicExtract
  Cce::AnalyticWorldtubeBoundary
  CharacteristicExtract
  )
add_cce_executable(
  KleinGordonCharacteristicExtract
  Cce::KleinGordonH5WorldtubeBoundary
  KleinGordonCharacteristicExtract
  )
