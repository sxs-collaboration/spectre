# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_REQUIRED_CHARM_VERSION 7.0.0)

option(USE_SCOTCH_LB "Use the charm++ ScotchLB module" OFF)

set(SCOTCHLB_COMPONENT "")
if (USE_SCOTCH_LB)
  find_package(Scotch REQUIRED)

  message(STATUS "Scotch libs: " ${SCOTCH_LIBRARIES})
  message(STATUS "Scotch incl: " ${SCOTCH_INCLUDE_DIR})
  message(STATUS "Scotch vers: " ${SCOTCH_VERSION})

  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "Scotch version: ${SCOTCH_VERSION}\n"
    )

  add_library(Scotch INTERFACE IMPORTED)
  set_property(TARGET Scotch
    APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIR})
  set_property(TARGET Scotch
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${SCOTCH_LIBRARIES})

  add_interface_lib_headers(
    TARGET Scotch
    HEADERS
    scotch.h
    )

  set(SCOTCHLB_COMPONENT ScotchLB)
endif()

find_package(Charm ${SPECTRE_REQUIRED_CHARM_VERSION} REQUIRED
  COMPONENTS
  EveryLB
  ${SCOTCHLB_COMPONENT}
  )
if(CHARM_VERSION VERSION_GREATER 7.0.0)
  message(WARNING "Builds with Charm++ versions greater than 7.0.0 are \
considered experimental. Please file any issues you encounter.")
endif()

if (USE_SCOTCH_LB)
  target_link_libraries(Charmxx::charmxx INTERFACE Scotch)
endif()

if(CHARM_TRACE_PROJECTIONS OR CHARM_TRACE_PROJECTIONS)
  set_property(TARGET SpectreFlags
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    -DSPECTRE_CHARM_PROJECTIONS
    -DSPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID=1000
    )
endif()

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Charm++ version: ${CHARM_VERSION}\n"
  "CHARM_COMPILER: ${CHARM_COMPILER}\n"
  "CHARM_SHARED_LIBS: ${CHARM_SHARED_LIBS}\n"
  "CHARM_BUILDING_BLOCKS:\n${CHARM_BUILDING_BLOCKS}\n"
  )

add_interface_lib_headers(
  TARGET
  Charmxx::charmxx
  HEADERS
  charm++.h
  )
add_interface_lib_headers(
  TARGET
  Charmxx::pup
  HEADERS
  pup.h
  pup_stl.h
  )
set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Charmxx::charmxx Charmxx::pup
  )

get_filename_component(CHARM_BINDIR ${CHARM_COMPILER} DIRECTORY)
# In order to avoid problems when compiling in parallel we manually copy the
# charmrun script over, rather than having charmc do it for us.
configure_file(
    "${CHARM_BINDIR}/charmrun"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/charmrun" COPYONLY
)

include(SetupCharmModuleFunctions)

# Make sure Charm++ was patched. If not you can get thread local storage errors
# when loading the python bindings.
if(NOT APPLE AND CHARM_VERSION VERSION_EQUAL 7.0.0 AND BUILD_PYTHON_BINDINGS)
  # Check that the patch was applied:
  set(CHARM_CHECK_FILE "${CHARM_INCLUDE_DIR}/conv-mach-opt.sh")
  if (EXISTS ${CHARM_CHECK_FILE})
    set(_CHARM_CMAKE_FILE_TO_CHECK ${CHARM_CHECK_FILE})
    file(READ ${_CHARM_CMAKE_FILE_TO_CHECK} FILE_CONTENTS)
    set(_EXPECTED_TLS_STRING "-ftls-model=initial-exec")
    string(FIND "${FILE_CONTENTS}"
      ${_EXPECTED_TLS_STRING} _LOCATION_OF_TLS)
    if (NOT ${_LOCATION_OF_TLS} EQUAL -1)
      message(FATAL_ERROR "Found -ftls-model=initial-exec flag. "
        "It looks like you forgot to apply the Charm++ patch. "
        "This is necessary when using the python bindings.")
    endif()
  else()
    message(STATUS "Unable to check if Charm++ was patched. "
      "Missing file ${CHARM_CHECK_FILE}")
  endif()
endif()
