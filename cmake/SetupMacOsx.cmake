# Distributed under the MIT License.
# See LICENSE.txt for details.

if(APPLE)
  # The -fvisibility=hidden flag is necessary to eliminate warnings
  # when building on Apple Silicon
  if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  endif()
  set(SPECTRE_MACOSX_MIN "10.9")
  if(DEFINED MACOSX_MIN)
    set(SPECTRE_MACOSX_MIN "${MACOSX_MIN}")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=${SPECTRE_MACOSX_MIN}")
  message(STATUS "Minimum macOS version: ${SPECTRE_MACOSX_MIN}")
endif()
