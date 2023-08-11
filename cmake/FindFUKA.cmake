# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find the FUKA initial data code (https://bitbucket.org/fukaws/fuka/src/fuka/)
#
# Pass `FUKA_ROOT` to the CMake build configuration to set up the following
# targets:
#
# - FUKA::Exporter: Functionality to load FUKA volume data and interpolate to
#   arbitrary points.

if(NOT FUKA_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(FUKA_ROOT "")
  set(FUKA_ROOT $ENV{FUKA_ROOT})
endif()

find_library(
  FUKA_LIB
  NAMES libkadath.a
  PATHS ${FUKA_ROOT}
  NO_DEFAULT_PATHS
  )

# Link MPI (should be the same MPI that FUKA was built with)
find_package(MPI COMPONENTS C)
find_package(FFTW)

if (FUKA_LIB AND MPI_C_FOUND AND FFTW_FOUND)
  add_library(FUKA::Exporter INTERFACE IMPORTED)
  target_link_libraries(
    FUKA::Exporter
    INTERFACE
    MPI::MPI_C
    FFTW::FFTW
    ${FUKA_LIB}
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  FUKA
  REQUIRED_VARS
  FUKA_LIB
  )
