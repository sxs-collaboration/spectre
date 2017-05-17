# Distributed under the MIT License.
# See LICENSE.txt for details.

option(ASAN "Add AddressSanitizer compile flags" OFF)
if (ASAN)
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer  -fsanitize=address"
  )
  set(
      CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address"
  )
endif ()

option(UBSAN_UNDEFINED "Add UBSan undefined behavior compile flags" OFF)
if (UBSAN_UNDEFINED)
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer  -fsanitize=undefined"
  )
  set(
      CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined"
  )
endif()

option(UBSAN_INTEGER "Add UBSan unsigned integer overflow compile flags" OFF)
if (UBSAN_INTEGER)
  set(
      CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer  -fsanitize=integer"
  )
  set(
      CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=integer"
  )
endif()
