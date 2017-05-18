# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(YAMLCPP REQUIRED)
include_directories(SYSTEM ${YAMLCPP_INCLUDE_DIRS})
list(APPEND SPECTRE_LIBRARIES ${YAMLCPP_LIBRARIES})

message(STATUS "yaml-cpp libs: " ${YAMLCPP_LIBRARIES})
message(STATUS "yaml-cpp incl: " ${YAMLCPP_INCLUDE_DIRS})
