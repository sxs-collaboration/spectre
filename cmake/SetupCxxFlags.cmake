# Distributed under the MIT License.
# See LICENSE.txt for details.

option(DEBUG_SYMBOLS "Add -g to CMAKE_CXX_FLAGS if ON, -g0 if OFF." ON)

option(OVERRIDE_ARCH "The architecture to use. Default is native." OFF)

option(ENABLE_SPECTRE_DEBUG "Enable ASSERTs and other SPECTRE_DEBUG options"
  OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ENABLE_SPECTRE_DEBUG ON)
endif()

if(${ENABLE_SPECTRE_DEBUG})
  set_property(TARGET SpectreFlags
    APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS SPECTRE_DEBUG)
endif()

if(APPLE AND "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
  # Because of a bug in macOS on Apple Silicon, executables larger than
  # 2GB in size cannot run. The -Oz flag minimizes executable size, to
  # avoid this bug.
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG -Oz")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG")
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

# Always compile only for the current architecture. This can be overridden
# by passing `-D OVERRIDE_ARCH=THE_ARCHITECTURE` to CMake
if(NOT "${OVERRIDE_ARCH}" STREQUAL "OFF")
  set_property(TARGET SpectreFlags
      APPEND PROPERTY
      INTERFACE_COMPILE_OPTIONS
      # The -mno-avx512f flag is necessary to avoid a Blaze 3.8 bug. The flag
      # should be re-enabled when we can insist on Blaze 3.9 which will include
      # a fix that allows this vectorization flag again.
      $<$<COMPILE_LANGUAGE:C>:-march=${OVERRIDE_ARCH} -mno-avx512f>
      $<$<COMPILE_LANGUAGE:CXX>:-march=${OVERRIDE_ARCH} -mno-avx512f>
      $<$<COMPILE_LANGUAGE:Fortran>:-march=${OVERRIDE_ARCH} -mno-avx512f>)
else()
  # Apple Silicon Macs do not support the -march flag or the -mno-avx512f flag
  if((NOT APPLE OR NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    # sometimes ARM architectures use the name "aarch64"
    AND NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64"
    )
    set_property(TARGET SpectreFlags
        APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:C>:-march=native -mno-avx512f>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native -mno-avx512f>
        $<$<COMPILE_LANGUAGE:Fortran>:-march=native -mno-avx512f>)
  endif()
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

# By default, the LLVM optimizer assumes floating point exceptions are ignored.
create_cxx_flag_target("-ffp-exception-behavior=maytrap" SpectreFpExceptions)
target_link_libraries(
  SpectreFlags
  INTERFACE
  SpectreFpExceptions
  )
