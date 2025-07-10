# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_KOKKOS "Use Kokkos" OFF)

if(SPECTRE_KOKKOS)
  find_package(Kokkos)

  if (Kokkos_FOUND)
    # Found external Kokkos installation, so just check that a few important
    # features are enabled
    if (Kokkos_ENABLE_CUDA)
      # Allow constexpr functions to be called from device code
      # without the need for a device annotation.
      # See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-functions-and-function-templates
      if (NOT Kokkos_ENABLE_CUDA_CONSTEXPR)
        message(FATAL_ERROR "Kokkos_ENABLE_CUDA_CONSTEXPR must be ON. "
          "Please recompile Kokkos with this option enabled.")
      endif()
      # Allow CUDA code in static libs, see
      # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
      # We may have to look into LTO if this has a performance penalty.
      if (NOT Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
        message(FATAL_ERROR "Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE "
          "must be ON. Please recompile Kokkos with this option enabled.")
      endif()
      if (BUILD_SHARED_LIBS)
        message(FATAL_ERROR "Building with shared libs is not supported when "
          "building with CUDA support. Set BUILD_SHARED_LIBS=OFF.")
      endif()
    endif()  # Kokkos_ENABLE_CUDA
  else()  # Kokkos_FOUND
    # Kokkos not found, so we will try to fetch it.
    if (NOT SPECTRE_FETCH_MISSING_DEPS)
      message(FATAL_ERROR "Could not find Kokkos. If you want to fetch "
        "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
    endif()

    # Configure the Kokkos build
    set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
      "Kokkos aggressive vectorization")

    if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR SPECTRE_DEBUG)
      message(STATUS "Enabling Kokkos debug mode")
      set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
      set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
        "Bounds checking on Kokkos views")
      set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
        "Sanity checks on Kokkos DualView")
    endif()

    if(Kokkos_ENABLE_CUDA)
      set(CMAKE_CUDA_STANDARD 20)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      enable_language(CUDA)
      find_package(CUDAToolkit REQUIRED)
      set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL
        "Enable constexpr in CUDA")
      set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL
        "Enable relocatable device code in CUDA")
      if (BUILD_SHARED_LIBS)
        message(FATAL_ERROR "Building with shared libs is not supported when "
          "building with CUDA support. Set BUILD_SHARED_LIBS=OFF.")
      endif()
    endif()

    message(STATUS "Fetching Kokkos")
    include(FetchContent)
    FetchContent_Declare(Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.4.00
      GIT_SHALLOW TRUE
      ${SPECTRE_FETCHCONTENT_BASE_ARGS}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

# Determine if the compiler is NVIDIA's nvcc (if not already determined by
# Kokkos)
if (NOT DEFINED KOKKOS_CXX_COMPILER_ID)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE _COMPILER_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" " " _COMPILER_VERSION ${_COMPILER_VERSION})
  string(FIND ${_COMPILER_VERSION} "nvcc" _COMPILER_IS_NVCC)
  if(${_COMPILER_IS_NVCC} GREATER -1)
    set(KOKKOS_CXX_COMPILER_ID "NVIDIA")
  else()
    set(KOKKOS_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
  endif()
endif()

# Include CUDA as system headers to suppress warnings. NVCC seems to include
# CUDA headers as `-I` instead of `-isystem` for some reason (as of
# version 12.6).
if (CUDAToolkit_INCLUDE_DIRS)
  create_cxx_flag_target(
    "-Xcompiler \"-isystem${CUDAToolkit_INCLUDE_DIRS}\""
    SpectreCudaSystemInclude)
  target_link_libraries(
    SpectreWarnings
    INTERFACE
    SpectreCudaSystemInclude
    )
endif()
