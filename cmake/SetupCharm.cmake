# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Charm 6.8.0 EXACT REQUIRED)

spectre_include_directories("${CHARM_INCLUDE_DIRS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L${CHARM_LIBRARIES}")

# SpECTRE must be linked with Charm++'s script charmc. In turn, charmc
# will call your normal compiler, set at charm++ installation time internally.
# Note: The -pthread is necessary with Charm v6.8 to get linking working
#       with GCC
string(
    REGEX REPLACE "<CMAKE_CXX_COMPILER>"
    "${CHARM_COMPILER} -pthread -module DistributedLB -no-charmrun"
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

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Charm Version:  ${CHARM_VERSION}\n"
  )
