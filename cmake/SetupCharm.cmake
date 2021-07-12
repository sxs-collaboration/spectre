# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_SCOTCH_LB "Use the charm++ ScotchLB module" OFF)

set(SCOTCHLB_COMPONENT "")
if (USE_SCOTCH_LB)
  set(SCOTCHLB_COMPONENT ScotchLB)
endif()

find_package(Charm 6.10.2 EXACT REQUIRED
  COMPONENTS
  EveryLB
  ${SCOTCHLB_COMPONENT}
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

set(SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD ${CHARM_SMP})
if (${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD})
  message(STATUS "Charm++ built with shared memory parallelism")
  add_definitions(-DSPECTRE_SHARED_MEMORY_PARALLELISM_BUILD)
else()
  message(STATUS "Charm++ NOT built with shared memory parallelism")
endif()

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Charm Version:  ${CHARM_VERSION}\n"
  "Charm SMP:  ${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD}\n"
  )
