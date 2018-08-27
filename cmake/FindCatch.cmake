# From: https://github.com/cinemast/libjson-rpc-cpp
# Copyright (C) 2011-2016 Peter Spiess-Knafl

# SpECTRE modifications:
# - add PATH_SUFFIXES to find_path
# - call function _get_catch_version (taken from another source)
# - pass version to find_package_handle_standard_args

# This function is from: https://github.com/pybind/pybind11
# Extract the version number from catch.hpp
function(_get_catch_version)
  file(
    STRINGS "${CATCH_INCLUDE_DIR}/catch.hpp"
    version_line REGEX "Catch v.*" LIMIT_COUNT 1
    )
  if(version_line MATCHES "Catch v([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(
      CATCH_VERSION
      "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" PARENT_SCOPE
      )
  endif()
endfunction()

find_path(
  CATCH_INCLUDE_DIR
  PATH_SUFFIXES single_include include catch catch2
  NAMES catch.hpp
  HINTS ${CATCH_ROOT}
  DOC "catch include dir"
  )

if(CATCH_INCLUDE_DIR)
  _get_catch_version()
endif()

set(CATCH_INCLUDE_DIRS ${CATCH_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Catch REQUIRED_VARS CATCH_INCLUDE_DIR VERSION_VAR CATCH_VERSION
  )

mark_as_advanced(CATCH_INCLUDE_DIR)
mark_as_advanced(CATCH_VERSION)
