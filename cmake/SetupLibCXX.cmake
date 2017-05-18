# Distributed under the MIT License.
# See LICENSE.txt for details.

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  find_package(LIBCXX REQUIRED)
  set(
      CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}\
 -L${LIBCXXABI_LIBRARIES} -L${LIBCXX_LIBRARIES} -stdlib=libc++ -lc++abi"
  )
  include_directories(${LIBCXX_INCLUDE_DIRS})
  set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
  message(STATUS "libc++ include: ${LIBCXX_INCLUDE_DIRS}")
  message(STATUS "libc++ libraries: ${LIBCXX_LIBRARIES}")
endif()
