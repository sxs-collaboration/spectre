# Distributed under the MIT License.
# See LICENSE.txt for details.

if(NOT USE_PAPI)
  option(USE_PAPI "Search for and include PAPI" OFF)
endif()

# Do not set up PAPI again if it was already done once
if(USE_PAPI AND NOT PAPI_FOUND)
  find_package(PAPI REQUIRED)
  spectre_include_directories(${PAPI_INCLUDE_DIRS})
  list(APPEND SPECTRE_LIBRARIES ${PAPI_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DSPECTRE_USE_PAPI")
endif()
