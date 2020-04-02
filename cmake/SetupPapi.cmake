# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT USE_PAPI)
  option(USE_PAPI "Search for and include PAPI" OFF)
endif()

# Do not set up PAPI again if it was already done once
if(USE_PAPI AND NOT TARGET Papi)
  find_package(PAPI REQUIRED)
  add_library(Papi INTERFACE IMPORTED)
  set_property(TARGET Papi
    APPEND PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${PAPI_INCLUDE_DIRS})
  set_property(TARGET Papi
    APPEND PROPERTY
    INTERFACE_COMPILE_DEFINITIONS "SPECTRE_USE_PAPI")
  set_property(TARGET Papi
    APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES ${PAPI_LIBRARIES})

  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    Papi
    )
endif(USE_PAPI AND NOT TARGET Papi)
