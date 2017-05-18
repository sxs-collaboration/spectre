# Distributed under the MIT License.
# See LICENSE.txt for details.

option(
    PROJECTIONS
    "Link flags to enable charm++ Projections analysis framework"
    OFF
)

if(NOT PROJECTIONS_PAPI_COUNTERS)
  option(
      PROJECTIONS_PAPI_COUNTERS
      "Comma separated list of PAPI counters to monitor. To see available \
 counters on your hardware run `$ papi_avail -a`"
      OFF
  )
endif()

if (NOT PROJECTIONS_USER_STATS)
  option(
      PROJECTIONS_USER_STATS
      "Register user-specified stats, requires Charm v6.8.0 or newer"
      OFF
  )
endif()

function(CHECK_CHARM_VERSION_FOR_STAT_TRACING)
  if(CHARM_MAJOR_VERSION AND CHARM_MINOR_VERSION)
    if(6 EQUAL CHARM_MAJOR_VERSION AND 8 GREATER CHARM_MINOR_VERSION)
      message(FATAL_ERROR
          "Stat tracing of PAPI counters not supported until Charm++ v6.8.0")
    endif()
  endif()
endfunction()

if (PROJECTIONS)
  set(
      CMAKE_CXX_LINK_FLAGS
      "${CMAKE_CXX_LINK_FLAGS} -tracemode projections -tracemode summary"
  )
  # The values 1000 and 1001 are chosen arbitrary but assume that there are
  # fewer than 1000 user-defined Actions (which should be a reasonable
  # assumption).
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} \
 -D SPECTRE_CHARM_PROJECTIONS \
 -D SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID=1000 \
 -D SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID=1001"
  )
  if (PROJECTIONS_PAPI_COUNTERS)
    check_charm_version_for_stat_tracing()

    set(USE_PAPI ON)
    string(
        REPLACE " " ""
        PROJECTIONS_PAPI_COUNTERS
        ${PROJECTIONS_PAPI_COUNTERS}
    )
    string(
        REPLACE "," ";"
        PROJECTIONS_PAPI_COUNTERS
        ${PROJECTIONS_PAPI_COUNTERS}
    )
    set(SPECTRE_PAPI_COUNTERS "")
    foreach(PAPI_COUNTER ${PROJECTIONS_PAPI_COUNTERS})
      set(
          SPECTRE_PAPI_COUNTERS
          "${SPECTRE_PAPI_COUNTERS}(${PAPI_COUNTER})"
      )
    endforeach()
    set(
        CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} \
-D SPECTRE_PAPI_COUNTERS=\"${SPECTRE_PAPI_COUNTERS}\""
    )
    include(SetupPapi)
  endif()
  if(PROJECTIONS_USER_STATS)
    CHECK_CHARM_VERSION_FOR_STAT_TRACING()
    set(
        CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -D SPECTRE_PROJECTIONS_USER_STATS"
    )
  endif()
endif ()
