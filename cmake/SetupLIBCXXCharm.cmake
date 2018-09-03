# Distributed under the MIT License.
# See LICENSE.txt for details.

# We need to add the link flag after finding packages because adding a
# compilation flag to the linker flags causes packages/libraries to not be
# always be found correctly.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND USE_LIBCXX)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
endif()
