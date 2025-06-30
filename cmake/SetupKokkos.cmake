# Distributed under the MIT License.
# See LICENSE.txt for details.

option(SPECTRE_KOKKOS "Use Kokkos" OFF)

if(SPECTRE_KOKKOS)
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
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL
      "Enable lambda expressions in CUDA")
    # Allow constexpr functions to be called from device code
    # without the need for a device annotation.
    # See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-functions-and-function-templates
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL
      "Enable constexpr in CUDA")
  endif()

  find_package(Kokkos)

  if (NOT Kokkos_FOUND)
    if (NOT SPECTRE_FETCH_MISSING_DEPS)
      message(FATAL_ERROR "Could not find Kokkos. If you want to fetch "
        "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
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
