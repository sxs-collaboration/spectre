# Distributed under the MIT License.
# See LICENSE.txt for details.

# Check whether the standard library is affected by
# https://bugs.llvm.org/show_bug.cgi?id=35491

message(STATUS "Checking for broken std::array<..., 0>")
try_compile(
  ARRAY0_WORKS
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/cmake/CheckBrokenArray0.cpp
  CMAKE_FLAGS
  -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
  )
if (ARRAY0_WORKS)
  message(STATUS "Checking for broken std::array<..., 0> -- works")
else (ARRAY0_WORKS)
  message(STATUS "Checking for broken std::array<..., 0> -- broken")
  add_definitions(-DHAVE_BROKEN_ARRAY0)
endif (ARRAY0_WORKS)
