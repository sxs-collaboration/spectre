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
  # Sets the macro `SPECTRE_PAPI_COUNTERS` to a list of user-specfied PAPI
  # counters (obtained from running `papi_avail`). The counters could be used
  # in conjunction with Charm++'s user stats tracing facilities. However,
  # care must be taken if/when this is implemented to not assume that an
  # Action will start and end without any Actions being run in the middle.
  # Because we sometimes elide calls to the Charm++ RTS when invoking
  # Actions, an Action on a local component element could be executed as part
  # of the current component element's execution. The PAPI counter stats
  # must be kept separate for these two Actions in order for them to be
  # meaningful.
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
  # Defines the macro SPECTRE_PROJECTIONS_USER_STATS if gathering user
  # specified stats is supported by the Charm++ version.
  if(PROJECTIONS_USER_STATS)
    CHECK_CHARM_VERSION_FOR_STAT_TRACING()
    add_definitions(-DSPECTRE_PROJECTIONS_USER_STATS)
  endif()
endif ()
