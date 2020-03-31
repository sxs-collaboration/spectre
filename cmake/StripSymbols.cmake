# Distributed under the MIT License.
# See LICENSE.txt for details.

option(STRIP_SYMBOLS "Strip symbols from executables" OFF)

if (STRIP_SYMBOLS)
  if (APPLE)
    message(FATAL_ERROR "Stripping all symbols is currently not supported on "
    "macOS. Please disable the `STRIP_SYMBOLS` flag.")
  else (APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--strip-all")
  endif (APPLE)
endif(STRIP_SYMBOLS)
