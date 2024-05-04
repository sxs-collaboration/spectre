# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Libsharp)

set(LIBSHARP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/external/libsharp/)

# Note:
# - libsharp is configured a bit oddly compared to spectre. In particular, there
#   are *_inc.c files that are included in other c files. These seem to be
#   similar in spirit to our .tpp files.
# - We do not compile sharp_testsuite.c since we don't need or want to test it.
# - We do not compile sharp_mpi.c since we don't want MPI usage in libsharp.
# - OpenMP support is explicitly commented out of libsharp since we do our own
#   threading in spectre.
# - We always build libsharp with -O3 so that everything runs fast.
set(LIBRARY_SOURCES
  bluestein.c
  c_utils.c
  fftpack.c
  ls_fft.c
  memusage.c
  sharp_almhelpers.c
  sharp_announce.c
  sharp.c
  sharp_core.c
  sharp_geomhelpers.c
  sharp_legendre.c
  sharp_legendre_roots.c
  sharp_legendre_table.c
  sharp_ylmgen_c.c
  walltime_c.c
)

add_library(
  ${LIBRARY}
  ${LIBRARY_SOURCES}
)

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  SpectreFlags
)

set_property(TARGET ${LIBRARY}
  APPEND PROPERTY
  COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:C>:-O3 -g0>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -g0>
  $<$<COMPILE_LANGUAGE:Fortran>:-O3 -g0>
)

target_include_directories(${LIBRARY} SYSTEM INTERFACE ${LIBSHARP_INCLUDE_DIR})
