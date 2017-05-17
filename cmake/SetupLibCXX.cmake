# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  find_package(LibCXX REQUIRED)
  set(
      CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}\
 -L${LIBCXXABI_LIBRARIES} -L${LIBCXX_LIBRARIES} -stdlib=libc++ -lc++abi"
  )
  include_directories(${LIBCXX_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
endif()
