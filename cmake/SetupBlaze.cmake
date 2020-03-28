# Distributed under the MIT License.
# See LICENSE.txt for details.

# Every time we've upgraded blaze compatibility in the past, we've had to change
# vector code, so we should expect to need changes again on each subsequent
# release, so we specify an exact version requirement.
find_package(Blaze 3.8 EXACT REQUIRED)

message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Blaze Version:  ${BLAZE_VERSION}\n"
  )

add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})
set_property(TARGET Blaze PROPERTY
  INTERFACE_LINK_LIBRARIES Lapack)

target_compile_definitions(Blaze
  INTERFACE
  # Override SMP configurations
  BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0
  BLAZE_OPENMP_PARALLEL_MODE=0
  BLAZE_CPP_THREADS_PARALLEL_MODE=0
  BLAZE_BOOST_THREADS_PARALLEL_MODE=0
  # Disable MPI parallelization
  BLAZE_MPI_PARALLEL_MODE=0
  # Disable HPX parallelization
  BLAZE_HPX_PARALLEL_MODE=0
  BLAZE_USE_PADDING=0
  # Enable non-temporal stores for cache optimization of large data structures
  BLAZE_USE_STREAMING=1
  BLAZE_USE_OPTIMIZED_KERNELS=1
  # Skip initializing default-constructed structures for fundamental types
  BLAZE_USE_DEFAULT_INITIALIZATON=0
  )

add_interface_lib_headers(
  TARGET Blaze
  HEADERS
  blaze/math/CustomVector.h
  blaze/math/DynamicMatrix.h
  blaze/math/DynamicVector.h
  blaze/system/Optimizations.h
  blaze/system/Version.h
  blaze/util/typetraits/RemoveConst.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blaze
  )
