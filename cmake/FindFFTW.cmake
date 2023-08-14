# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find FFTW3 (The Fastest Fourier Transform in the West)
# http://www.fftw.org/
#
# Defines the `FFTW::FFTW` target to link against.

if(NOT FFTW_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(FFTW_ROOT "")
  set(FFTW_ROOT $ENV{FFTW_ROOT})
endif()

find_path(FFTW_INCLUDE_DIR
  NAMES fftw3.h
  PATH_SUFFIXES include
  HINTS ${FFTW_ROOT}
  )

find_library(FFTW_LIB
  NAMES fftw3
  PATH_SUFFIXES lib64 lib
  HINTS ${FFTW_ROOT}
  )

if (FFTW_INCLUDE_DIR AND FFTW_LIB)
  add_library(FFTW::FFTW INTERFACE IMPORTED)
  target_include_directories(FFTW::FFTW INTERFACE ${FFTW_INCLUDE_DIR})
  target_link_libraries(FFTW::FFTW INTERFACE ${FFTW_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW
  REQUIRED_VARS FFTW_INCLUDE_DIR FFTW_LIB)
mark_as_advanced(FFTW_INCLUDE_DIR FFTW_LIB)
