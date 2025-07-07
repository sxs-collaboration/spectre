# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(brigand)
add_subdirectory(EffectiveSource)
add_subdirectory(SPHEREPACK)
add_subdirectory(slatec)

set(LIBRARY Libsharp)
set(LIBSHARP_LIBDIR ${CMAKE_CURRENT_BINARY_DIR}/libsharp/auto/lib)
set(LIBSHARP_LIB_libsharp ${LIBSHARP_LIBDIR}/libsharp.a)
set(LIBSHARP_LIB_libfftpack ${LIBSHARP_LIBDIR}/libfftpack.a)
set(LIBSHARP_LIB_libc_utils ${LIBSHARP_LIBDIR}/libc_utils.a)
set(LIBSHARP_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/libsharp/auto/include)

find_program(MAKE NAMES gmake make REQUIRED)
set(LIBSHARP_BUILD ${MAKE})
# When using the Unix Makefile generator, verbosity is inherited from
# the main build and works acceptably well.  There's no way to
# dynamically detect the verbosity with ninja, so silence it
# unconditionally because ninja is quieter than make.
if (NOT CMAKE_GENERATOR STREQUAL "Unix Makefiles")
  list(APPEND LIBSHARP_BUILD > /dev/null)
endif()

set(JOB_SERVER "")
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
  set(JOB_SERVER BUILD_JOB_SERVER_AWARE TRUE)
endif()

include(ExternalProject)
ExternalProject_Add(
  Libsharp-external
  PREFIX ${CMAKE_BINARY_DIR}/external/libsharp
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/libsharp
  DOWNLOAD_COMMAND
  cp -r ${CMAKE_CURRENT_SOURCE_DIR}/libsharp ${CMAKE_CURRENT_BINARY_DIR}
  # libsharp has an autoconf build system, but it doesn't do anything
  # except set CFLAGS, so we can skip it to avoid depending on
  # autoconf and set the flags below by manually writing config.auto.
  CONFIGURE_COMMAND ""
  # out-of-tree builds don't work
  BUILD_IN_SOURCE TRUE
  BUILD_COMMAND "${LIBSHARP_BUILD}"
  ${JOB_SERVER}
  BUILD_BYPRODUCTS
  ${LIBSHARP_LIB_libsharp}
  ${LIBSHARP_LIB_libfftpack}
  ${LIBSHARP_LIB_libc_utils}
  INSTALL_COMMAND ""
)

# Always build libsharp with optimization, since there is a big speed
# difference and we're not interested in debugging it.
file(
  GENERATE OUTPUT "libsharp/config/config.auto"
  CONTENT
  "CC=${CMAKE_C_COMPILER}
CL=\$(CC)
SPECTRE_FLAGS=\\
\$(subst ;, ,$<TARGET_PROPERTY:SpectreFlags,INTERFACE_COMPILE_OPTIONS>)
CCFLAGS=\$(SPECTRE_FLAGS) -fno-openmp -O3 -c
CLFLAGS=-L. -L\$(LIBDIR) \$(SPECTRE_FLAGS) -fno-openmp -O3 -lm
ARCREATE=ar cr"
  CONDITION "$<COMPILE_LANGUAGE:C>"
)

add_library(Libsharp::libsharp STATIC IMPORTED GLOBAL)
set_target_properties(
  Libsharp::libsharp
  PROPERTIES
  IMPORTED_LOCATION
  ${LIBSHARP_LIB_libsharp}
)
add_library(Libsharp::libfftpack STATIC IMPORTED GLOBAL)
set_target_properties(
  Libsharp::libfftpack
  PROPERTIES
  IMPORTED_LOCATION
  ${LIBSHARP_LIB_libfftpack}
)
add_library(Libsharp::libc_utils STATIC IMPORTED GLOBAL)
set_target_properties(
  Libsharp::libc_utils
  PROPERTIES
  IMPORTED_LOCATION
  ${LIBSHARP_LIB_libc_utils}
)
add_library(Libsharp INTERFACE IMPORTED GLOBAL)
target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Libsharp::libsharp
  Libsharp::libfftpack
  Libsharp::libc_utils
)
# cmake issue #15052
file(MAKE_DIRECTORY ${LIBSHARP_INCLUDE})
target_include_directories(
  ${LIBRARY}
  SYSTEM
  INTERFACE
  ${LIBSHARP_INCLUDE}
)
add_dependencies(
  ${LIBRARY}
  Libsharp-external
)
