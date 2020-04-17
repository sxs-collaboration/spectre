# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find yaml-cpp: https://github.com/jbeder/yaml-cpp
# If not in one of the default paths specify -D YAMLCPP_ROOT=/path/to/yaml-cpp
# to search there as well.
# Static libraries can be use by setting -D YAMLCPP_STATIC_LIBRARY=ON

include (CheckCXXSourceRuns)

if(NOT YAMLCPP_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(YAMLCPP_ROOT "")
  set(YAMLCPP_ROOT $ENV{YAMLCPP_ROOT})
endif()

# find the yaml-cpp include directory
find_path(YAMLCPP_INCLUDE_DIRS yaml-cpp/yaml.h
    PATH_SUFFIXES include
    HINTS ${YAMLCPP_ROOT}/include/)

if(YAMLCPP_STATIC_LIBRARY)
  set(YAMLCPP_STATIC libyaml-cpp.a)
else(YAMLCPP_STATIC_LIBRARY)
  # Silence CMake uninitialized variable warning
  set(YAMLCPP_STATIC "")
endif()

find_library(YAMLCPP_LIBRARIES
    NAMES ${YAMLCPP_STATIC} yaml-cpp
    PATH_SUFFIXES lib64 lib
    HINTS ${YAMLCPP_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(YAMLCPP
    DEFAULT_MSG YAMLCPP_INCLUDE_DIRS YAMLCPP_LIBRARIES)
mark_as_advanced(YAMLCPP_INCLUDE_DIRS YAMLCPP_LIBRARIES)
