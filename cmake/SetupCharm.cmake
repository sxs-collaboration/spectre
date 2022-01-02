# Distributed under the MIT License.
# See LICENSE.txt for details.

# Apple Silicon (arm64) Macs require charm 7.0.0 to run
if(APPLE AND "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
  find_package(Charm 7.0.0 REQUIRED)
else()
  find_package(Charm 6.10.2 REQUIRED)
endif()
if(CHARM_VERSION VERSION_GREATER 6.10.2)
  message(WARNING "Builds with Charm++ versions greater than 6.10.2 are \
considered experimental. Please file any issues you encounter.")
endif()

spectre_include_directories("${CHARM_INCLUDE_DIRS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L${CHARM_LIBRARIES}")

if(NOT TARGET Charmxx)
  add_library(Charmxx INTERFACE IMPORTED)
  set_property(TARGET Charmxx PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${CHARM_INCLUDE_DIRS})
  add_interface_lib_headers(
    TARGET Charmxx
    HEADERS
    pup.h
    pup_stl.h
    )
  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    Charmxx
    )
endif(NOT TARGET Charmxx)

# SpECTRE must be linked with Charm++'s script charmc. In turn, charmc
# will call your normal compiler, set at charm++ installation time internally.
# Note: The -pthread is necessary with Charm v6.10 to get linking working
#       with GCC
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
    APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIRS})
  set_property(TARGET Scotch
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${SCOTCH_LIBRARIES})

  add_interface_lib_headers(
    TARGET Scotch
    HEADERS
    scotch.h
    )
  # We can't actually use the Scotch target until we have a Charm++ CMake
  # target. We can then override the `-lscotch` and `-lscotcherr` libs specified
  # in charmc and instead have CMake handle the dependency. Until then we
  # extract the directory in which we found Scotch and append that to the global
  # list of directories.
  list(GET SCOTCH_LIBRARIES 0 SCOTCH_LINK_LIB)
  get_filename_component(SCOTCH_LINK_LIB "${SCOTCH_LINK_LIB}" DIRECTORY)
  link_directories(BEFORE "${SCOTCH_LINK_LIB}")

  set(SCOTCH_LB_FLAG "-module ScotchLB")
else()
  set(SCOTCH_LB_FLAG "")
endif()

string(
    REGEX REPLACE "<CMAKE_CXX_COMPILER>"
    "${CHARM_COMPILER} -pthread -module EveryLB ${SCOTCH_LB_FLAG} -no-charmrun"
    CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")

# When building for trace analysis the PAPI counters passed to charmc
# aren't handled correctly. Specifically, the quotation marks are stripped
# while being passed through charmc. Since compilation flags aren't needed
# for linking, we strip them.
string(
    REGEX REPLACE "<FLAGS>" ""
    CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")

get_filename_component(CHARM_BINDIR ${CHARM_COMPILER} DIRECTORY)
# In order to avoid problems when compiling in parallel we manually copy the
# charmrun script over, rather than having charmc do it for us.
configure_file(
    "${CHARM_BINDIR}/charmrun"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/charmrun" COPYONLY
)

include(SetupCharmModuleFunctions)

# Check if Charm++ was built with SMP support
set(SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD NO)
if (EXISTS "${CHARM_INCLUDE_DIRS}/conv-mach.h")
  file(READ "${CHARM_INCLUDE_DIRS}/conv-mach.h" CONV_MACH_HEADER)
  string(REPLACE "\n" ";" CONV_MACH_HEADER ${CONV_MACH_HEADER})
  foreach (LINE ${CONV_MACH_HEADER})
    if (LINE MATCHES "#define[ \t]+CMK_SMP[ \t]+1")
      set(SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD YES)
      break()
    endif()
  endforeach()
else()
  message(FATAL_ERROR
    "Unable to detect whether or not Charm++ was built with shared "
    "memory parallelism enabled because the file "
    "'${CHARM_INCLUDE_DIRS}/conv-mach.h' was not found. Please file "
    "an issue for support with this error since that file should be "
    "generated as part of the Charm++ build process.")
endif()
if (${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD})
  message(STATUS "Charm++ built with shared memory parallelism")
  add_definitions(-DSPECTRE_SHARED_MEMORY_PARALLELISM_BUILD)
else()
  message(STATUS "Charm++ NOT built with shared memory parallelism")
endif()

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Charm version: ${CHARM_VERSION}\n"
  "Charm SMP: ${SPECTRE_SHARED_MEMORY_PARALLELISM_BUILD}\n"
  )
