# Distributed under the MIT License.
# See LICENSE.txt for details.

if(APPLE)
  set(SPECTRE_MACOSX_MIN "10.7")
  if(DEFINED MACOSX_MIN)
    set(SPECTRE_MACOSX_MIN "${MACOSX_MIN}")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=${SPECTRE_MACOSX_MIN}")
endif()
