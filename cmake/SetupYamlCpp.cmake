# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(YAMLCPP REQUIRED)

message(STATUS "yaml-cpp libs: " ${YAMLCPP_LIBRARIES})
message(STATUS "yaml-cpp incl: " ${YAMLCPP_INCLUDE_DIRS})

add_library(YamlCpp INTERFACE IMPORTED)
# Have to use `set_property` before CMake version 3.11. See:
# https://gitlab.kitware.com/cmake/cmake/-/merge_requests/1264
set_property(TARGET YamlCpp
  APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${YAMLCPP_INCLUDE_DIRS})
set_property(TARGET YamlCpp
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${YAMLCPP_LIBRARIES})

add_interface_lib_headers(
  TARGET YamlCpp
  HEADERS
  yaml-cpp/yaml.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  YamlCpp
  )
