# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find the COCAL initial data code (assumed repository or documentation link)
#
# Pass `COCAL_ROOT` to the CMake build configuration to set up the following
# targets:
#
# - COCAL::Exporter: Functionality to load COCAL volume data and interpolate to
#   arbitrary points.

if(NOT COCAL_ROOT)
  set(COCAL_ROOT "")
  set(COCAL_ROOT $ENV{COCAL_ROOT})
endif()

find_library(
  COCAL_LIB
  NAMES libcocal.a
  PATHS ${COCAL_ROOT}/lib
  NO_DEFAULT_PATHS
)

find_path(
  COCAL_INCLUDE_DIR
  NAMES coc2cac_bns.f90
  PATHS ${COCAL_ROOT}/include
  NO_DEFAULT_PATHS
)

# Optionally link MPI if COCAL is built with MPI dependencies
find_package(MPI COMPONENTS CXX)

if (COCAL_LIB AND MPI_CXX_FOUND)
  add_library(COCAL::Exporter INTERFACE IMPORTED)
  set_target_properties(COCAL::Exporter PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${COCAL_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${COCAL_LIB};MPI::MPI_CXX;${OtherDependency_LIBRARIES}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  COCAL
  FOUND_VAR COCAL_FOUND
  REQUIRED_VARS COCAL_LIB COCAL_INCLUDE_DIR
)
