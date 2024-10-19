# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find the COCAL initial data code (assumed repository or documentation link)
#
# Pass `COCAL_ROOT` to the CMake build configuration to set up the following
# targets:
#
# - COCAL::Exporter: Functionality to load COCAL volume data and interpolate to
#   arbitrary points.

if(NOT DEFINED COCAL_ROOT OR COCAL_ROOT STREQUAL "")
  set(COCAL_ROOT $ENV{COCAL_ROOT})
endif()

if (DEFINED COCAL_ROOT AND NOT COCAL_ROOT STREQUAL "")
  
  # Find the COCAL library
  find_library(
    COCAL_LIB
    NAMES libcocal.a
    PATHS ${COCAL_ROOT}/lib
    NO_DEFAULT_PATHS
  )

  # Find the COCAL include directory
  find_path(
    COCAL_INCLUDE_DIR
    NAMES coc2cac_bns.f90
    PATHS ${COCAL_ROOT}/include
    NO_DEFAULT_PATHS
  )

  # Find MPI
  find_package(MPI COMPONENTS CXX)

  if (COCAL_LIB AND MPI_CXX_FOUND)
    add_library(COCAL::Exporter INTERFACE IMPORTED)
    set_target_properties(COCAL::Exporter PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${COCAL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${COCAL_LIB};MPI::MPI_CXX"
    )
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    COCAL
    FOUND_VAR COCAL_FOUND
    REQUIRED_VARS COCAL_LIB COCAL_INCLUDE_DIR
  )
else()
  set(COCAL_FOUND FALSE)
endif()
