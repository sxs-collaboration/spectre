# Distributed under the MIT License.
# See LICENSE.txt for details.

include(SpectreCheckTargetDependencies)

function(check_spectre_libs_dependencies)
  get_property(
    SPECTRE_TPLS GLOBAL PROPERTY SPECTRE_THIRD_PARTY_LIBS
    )

  get_property(
    SPECTRE_LIBS
    TARGET libs
    PROPERTY MANUALLY_ADDED_DEPENDENCIES
    )

  foreach(TARGET_TO_CHECK ${ARGN})
    check_target_dependencies(
      TARGET ${TARGET_TO_CHECK}
      ALL_TARGETS
      ${SPECTRE_TPLS}
      ${SPECTRE_LIBS}
      ALLOWED_EXTRA_TARGETS
      SpectreFlags
      ERROR_ON_FAILURE
      )
  endforeach(TARGET_TO_CHECK ${ARGN})
endfunction(check_spectre_libs_dependencies)

option(CHECK_LIBRARY_DEPENDENCIES
  "Check link dependencies of SpECTRE libraries" OFF)

option(CHECK_ALL_DEPENDENCIES
  "Check link dependencies of all SpECTRE targets" OFF)

if(CHECK_LIBRARY_DEPENDENCIES OR CHECK_ALL_DEPENDENCIES)
  get_property(
    SPECTRE_LIBS
    TARGET libs
    PROPERTY MANUALLY_ADDED_DEPENDENCIES
    )
  check_spectre_libs_dependencies(
    ${SPECTRE_LIBS}
    )
endif(CHECK_LIBRARY_DEPENDENCIES OR CHECK_ALL_DEPENDENCIES)
