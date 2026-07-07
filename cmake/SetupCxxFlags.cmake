# Distributed under the MIT License.
# See LICENSE.txt for details.

option(DEBUG_SYMBOLS "Add -g to CMAKE_CXX_FLAGS if ON, -g0 if OFF." ON)

option(OVERRIDE_ARCH "The architecture to use. Default is native." OFF)

option(SPECTRE_DEBUG "Enable ASSERTs and other SPECTRE_DEBUG options"
  OFF)

# Because of a bug in macOS on Apple Silicon, executables larger than 2GB in
# size cannot run. This option minimizes executable size to avoid this bug. It
# is enabled by default on Apple Silicon.
set(_SPECTRE_OPTIMIZE_SIZE_DEFAULT OFF)
if(APPLE AND "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
  set(_SPECTRE_OPTIMIZE_SIZE_DEFAULT ON)
endif()
option(SPECTRE_OPTIMIZE_SIZE "Optimize for executable size instead of speed"
  ${_SPECTRE_OPTIMIZE_SIZE_DEFAULT})

option(SPECTRE_DEBUG_Og "Compile Debug builds with -Og instead of -O0" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(SPECTRE_DEBUG ON)
endif()

if(${SPECTRE_DEBUG})
  set_property(TARGET SpectreFlags
    APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS SPECTRE_DEBUG)
endif()

option(SPECTRE_NAN_INIT "Initialize memory to NaN is various places"
  ${SPECTRE_DEBUG})

if(${SPECTRE_NAN_INIT})
  set_property(TARGET SpectreFlags
    APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS SPECTRE_NAN_INIT)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")

if(${SPECTRE_OPTIMIZE_SIZE})
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Oz")
elseif(${SPECTRE_DEBUG_Og})
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -Og")
endif()

if(NOT ${DEBUG_SYMBOLS})
  string(REPLACE "-g " "-g0 " CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

# Always build with -g so we can view backtraces, etc. when production code
# fails. This can be overridden by passing `-D DEBUG_SYMBOLS=OFF` to CMake
if(${DEBUG_SYMBOLS})
  set_property(TARGET SpectreFlags
    APPEND PROPERTY INTERFACE_COMPILE_OPTIONS -g)
endif(${DEBUG_SYMBOLS})

set(_NO_AVX512 "")
if((NOT APPLE OR NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    # sometimes ARM architectures use the name "aarch64"
    AND NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64"
  )
  # The -mno-avx512f flag is necessary to avoid a Blaze 3.8 bug. The flag
  # should be re-enabled when we can insist on Blaze 3.9 which will include
  # a fix that allows this vectorization flag again.
  set(_NO_AVX512 "-mno-avx512f")
endif()

# Check whether -march=native is supported before adding it to SpectreFlags.
# This avoids using the flag on compilers that reject it, such as clang on
# arm64 Linux.
function(_spectre_check_native_architecture_flag LANGUAGE XTYPE SOURCE_FILE
         FLAG_VARIABLE)
  set(${FLAG_VARIABLE} "" PARENT_SCOPE)
  execute_process(
    COMMAND
    bash -c
    "LC_ALL=POSIX ${CMAKE_${LANGUAGE}_COMPILER} -Werror \
-march=native -x ${XTYPE} -c ${SOURCE_FILE} -o /dev/null"
    RESULT_VARIABLE RESULT
    ERROR_VARIABLE ERROR_FROM_COMPILATION
    OUTPUT_QUIET)
  if(${RESULT} EQUAL 0)
    set(${FLAG_VARIABLE} "-march=native" PARENT_SCOPE)
  endif()
endfunction()

# Always compile only for the current architecture. This can be overridden
# by passing `-D OVERRIDE_ARCH=THE_ARCHITECTURE` to CMake
if(NOT "${OVERRIDE_ARCH}" STREQUAL "OFF")
  if(NOT APPLE)
    set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:C>:-march=${OVERRIDE_ARCH} ${_NO_AVX512}>
      $<$<COMPILE_LANGUAGE:CXX>:-march=${OVERRIDE_ARCH} ${_NO_AVX512}>
      $<$<COMPILE_LANGUAGE:Fortran>:-march=${OVERRIDE_ARCH} ${_NO_AVX512}>)
  endif()
else()
  _spectre_check_native_architecture_flag(
    C c ${_CHECK_CXX_FLAGS_SOURCE} _SPECTRE_C_NATIVE_ARCHITECTURE_FLAG)
  _spectre_check_native_architecture_flag(
    CXX c++ ${_CHECK_CXX_FLAGS_SOURCE} _SPECTRE_CXX_NATIVE_ARCHITECTURE_FLAG)
  _spectre_check_native_architecture_flag(
    Fortran f95 ${_CHECK_FORTRAN_FLAGS_SOURCE}
    _SPECTRE_FORTRAN_NATIVE_ARCHITECTURE_FLAG)
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:C>:${_SPECTRE_C_NATIVE_ARCHITECTURE_FLAG}>
      $<$<COMPILE_LANGUAGE:C>:${_NO_AVX512}>
      $<$<COMPILE_LANGUAGE:CXX>:${_SPECTRE_CXX_NATIVE_ARCHITECTURE_FLAG}>
      $<$<COMPILE_LANGUAGE:CXX>:${_NO_AVX512}>
      $<$<COMPILE_LANGUAGE:Fortran>:${_SPECTRE_FORTRAN_NATIVE_ARCHITECTURE_FLAG}>
      $<$<COMPILE_LANGUAGE:Fortran>:${_NO_AVX512}>)
endif()

# We are getting multiple types of linker warnings on macOS:
# - "ranlib: archive library: tmp/libSpectrePchLib.a the table of contents is
#   empty (no object file members in the library define global symbols)":
#   Yes, some of our libs have no symbols. Doesn't seem like a problem.
# - "-undefined dynamic_lookup may not work with chained fixups":
#   This warning appears when compiling Python bindings. Chained fixups were
#   introduced in AppleClang 13 and enabled by default in macOS 12. See these
#   upstream issues:
#   - CPython: https://github.com/python/cpython/issues/97524
#   - Pybind11: https://github.com/pybind/pybind11/pull/4301
#   - CMake: https://gitlab.kitware.com/cmake/cmake/-/issues/24044
#   Disabling chained fixups with `-Wl-no_fixup_chains` leads to linker warnings
#   about inconsistent visibility settings in different translation units. We
#   probably have to wait for an upstream solution to this issue.
# - "could not create compact unwind for SYMBOL: registers X and
#   Y not saved contiguously in frame":
#   We have seen these warnings on Apple Silicon chips.
#   Disabling compact unwind with the flags
#     -Wl,-keep_dwarf_unwind
#     -Wl,-no_compact_unwind
#   seems to work on some machines, but leads to segfaults on others. We haven't
#   investigated this in any more detail.
# For now we just suppress these linker warnings altogether, since we haven't
# encountered any problems with them and some are upstream issues.
if(APPLE)
  target_link_options(
    SpectreFlags
    INTERFACE
    -Wl,-w
    )
endif()

# We always want a detailed backtrace of template errors to make debugging them
# easier
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=0>)

# Increase bracket depth for fold expressions
# (see also https://github.com/llvm/llvm-project/issues/48973)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
  set_property(TARGET SpectreFlags
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:-fbracket-depth=1024>)
endif()

# Disable cmath setting the error flag. This allows the compiler to more
# aggressively vectorize code since it doesn't need to respect some global
# state.
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:C>:-fno-math-errno>
  $<$<COMPILE_LANGUAGE:CXX>:-fno-math-errno>
  $<$<COMPILE_LANGUAGE:Fortran>:-fno-math-errno>)

# Allow the compiler to transform divisions into multiplication by the
# reciprocal.
set_property(TARGET SpectreFlags
  APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:C>:-freciprocal-math>
  $<$<COMPILE_LANGUAGE:CXX>:-freciprocal-math>
  $<$<COMPILE_LANGUAGE:Fortran>:-freciprocal-math>)

# -ffp-exception-behavior=maytrap - By default, the LLVM optimizer assumes
#     floating point exceptions are ignored.
# -fnon-call-exceptions - By default, GCC does not allow signal handlers to
#     throw exceptions.
# Note: -ffp-exception-behavior is not supported on ARM64 architectures
if((NOT APPLE OR NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
  AND NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64"
  )
  create_c_flags_target(
    "-ffp-exception-behavior=maytrap;-fnon-call-exceptions"
    SpectreFpExceptions
    )
  create_cxx_flags_target(
    "-ffp-exception-behavior=maytrap;-fnon-call-exceptions"
    SpectreFpExceptions
    )
  target_link_libraries(
    SpectreFlags
    INTERFACE
    SpectreFpExceptions
    )
else()
  # On ARM64, we can't trap FP exceptions, so just enable non-call exceptions
  create_c_flags_target(
    "-fnon-call-exceptions"
    SpectreFpExceptions
    )
  create_cxx_flags_target(
    "-fnon-call-exceptions"
    SpectreFpExceptions
    )
  target_link_libraries(
    SpectreFlags
    INTERFACE
    SpectreFpExceptions
    )
endif()

file(APPEND
  "${CMAKE_BINARY_DIR}/BuildInfo.txt"
  "SPECTRE_DEBUG: ${SPECTRE_DEBUG}\n"
  "SPECTRE_NAN_INIT: ${SPECTRE_NAN_INIT}\n"
  )
