# Distributed under the MIT License.
# See LICENSE.txt for details.

option(USE_SLEEF "Use Sleef to add more vectorized instructions." OFF)

if(USE_SLEEF)
  # Try to find Sleef to increase vectorization
  find_package(Sleef)
endif()

if(SLEEF_FOUND)
  message(STATUS "Sleef libs: ${SLEEF_LIBRARIES}")
  message(STATUS "Sleef incl: ${SLEEF_INCLUDE_DIR}")
  message(STATUS "Sleef vers: ${SLEEF_VERSION}")

  file(APPEND
    "${CMAKE_BINARY_DIR}/BuildInfo.txt"
    "Sleef version: ${SLEEF_VERSION}\n"
  )
endif()

# Every time we've upgraded blaze compatibility in the past, we've had to change
# vector code, so we should expect to need changes again on each subsequent
# release, so we should specify an exact version requirement. However, Blaze
# hasn't been consistent in naming releases (version 3.8.2 has 3.9.0 written
# in Version.h).
find_package(Blaze 3.8 REQUIRED)

message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "Blaze version: ${BLAZE_VERSION}\n"
  )

add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})
target_link_libraries(
  Blaze
  INTERFACE
  Blas
  GSL::gsl # for BLAS header
  Lapack
  )
set(_BLAZE_USE_SLEEF 0)

if(SLEEF_FOUND)
  target_link_libraries(
    Blaze
    INTERFACE
    Sleef
    )
  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    Sleef
    )
  set(_BLAZE_USE_SLEEF 1)
endif()

# If BLAZE_USE_STRONG_INLINE=ON, Blaze will use this keyword to increase the
# likelihood of inlining. If BLAZE_USE_STRONG_INLINE=OFF, uses inline keyword
# as a fallback.
option(BLAZE_USE_STRONG_INLINE "Increase likelihood of Blaze inlining." ON)

set(_BLAZE_USE_STRONG_INLINE 0)

if(BLAZE_USE_STRONG_INLINE)
  set(_BLAZE_USE_STRONG_INLINE 1)
endif()

# If BLAZE_USE_ALWAYS_INLINE=ON, Blaze will use this keyword to force inlining.
# If BLAZE_USE_ALWAYS_INLINE=OFF or if the platform being used cannot 100%
# guarantee inlining, uses BLAZE_STRONG_INLINE as a fallback.
option(BLAZE_USE_ALWAYS_INLINE "Force Blaze inlining." ON)

set(_BLAZE_USE_ALWAYS_INLINE 0)

if(BLAZE_USE_ALWAYS_INLINE)
  set(_BLAZE_USE_ALWAYS_INLINE 1)
endif()

# Configure Blaze. Some of the Blaze configuration options could be optimized
# for the machine we are running on. See documentation:
# https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!step-2-configuration
target_compile_definitions(Blaze
  INTERFACE
  # - Enable external BLAS kernels
  BLAZE_BLAS_MODE=1
  # - Use BLAS header from GSL. We could also find and include a <cblas.h> (or
  #   similarly named) header that may be distributed with the BLAS
  #   implementation, but it's not guaranteed to be available and may conflict
  #   with the GSL header. Since we use GSL anyway, it's easier to use their
  #   BLAS header.
  BLAZE_BLAS_INCLUDE_FILE=<gsl/gsl_cblas.h>
  # - Set default matrix storage order to column-major, since many of our
  #   functions are implemented for column-major layout. This default reduces
  #   conversions.
  BLAZE_DEFAULT_STORAGE_ORDER=blaze::columnMajor
  # - Disable SMP parallelization. This disables SMP parallelization for all
  #   possible backends (OpenMP, C++11 threads, Boost, HPX):
  #   https://bitbucket.org/blaze-lib/blaze/wiki/Serial%20Execution#!option-3-deactivation-of-parallel-execution
  BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0
  # - Disable MPI parallelization
  BLAZE_MPI_PARALLEL_MODE=0
  # - Using the default cache size, which may have been configured automatically
  #   by the Blaze CMake configuration for the machine we are running on. We
  #   could override it here explicitly to tune performance.
  # BLAZE_CACHE_SIZE
  BLAZE_USE_PADDING=0
  # - Always enable non-temporal stores for cache optimization of large data
  #   structures: https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20Files#!streaming-non-temporal-stores
  BLAZE_USE_STREAMING=1
  # - Skip initializing default-constructed structures for fundamental types
  BLAZE_USE_DEFAULT_INITIALIZATON=0
  # Use Sleef for vectorization of more math functions
  BLAZE_USE_SLEEF=${_BLAZE_USE_SLEEF}
  # Set inlining settings
  BLAZE_USE_STRONG_INLINE=${_BLAZE_USE_STRONG_INLINE}
  BLAZE_USE_ALWAYS_INLINE=${_BLAZE_USE_ALWAYS_INLINE}
  )

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(Blaze
    INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include csignal>")
  # CMake doesn't like function macros in target_compile_definitions, so we
  # have to define it separately. We also need to make sure csignal is
  # included.
  target_compile_options(Blaze
    INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:
    -D 'BLAZE_THROW(EXCEPTION)=struct sigaction handler{}\;handler.sa_handler=\
SIG_IGN\;handler.sa_flags=0\;sigemptyset(&handler.sa_mask)\;\
sigaction(SIGTRAP,&handler,nullptr)\;raise(SIGTRAP)\;throw EXCEPTION'
    >")
else()
  # In release mode disable checks completely.
  set_property(TARGET Blaze
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:
    -D 'BLAZE_THROW(EXCEPTION)='
    >")
endif()


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
