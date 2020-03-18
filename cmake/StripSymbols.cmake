# Distributed under the MIT License.
# See LICENSE.txt for details.

option(STRIP_SYMBOLS "Strip symbols from executables" OFF)

if (STRIP_SYMBOLS)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--strip-all")
endif(STRIP_SYMBOLS)
