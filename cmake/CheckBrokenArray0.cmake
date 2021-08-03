# Distributed under the MIT License.
# See LICENSE.txt for details.

# Check whether the standard library is affected by
# https://bugs.llvm.org/show_bug.cgi?id=35491
# This LLVM bug is fixed by commit 59cdf90ac8bea16abbb9d637c5124e69d2c75c09,
# which is included in the LLVM 7 release.

message(STATUS "Checking for broken std::array<..., 0>")
try_compile(
  ARRAY0_WORKS
  ${CMAKE_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/cmake/CheckBrokenArray0.cpp
  CMAKE_FLAGS
  -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
  -DCMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}
  -DCMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}
  OUTPUT_VARIABLE TRY_COMPILE_OUTPUT
  )
if (ARRAY0_WORKS)
  message(STATUS "Checking for broken std::array<..., 0> -- works")
else (ARRAY0_WORKS)
  message(STATUS "Checking for broken std::array<..., 0> -- broken")
  add_definitions(-DHAVE_BROKEN_ARRAY0)
  message(STATUS "Output when trying to compile broken array test:\n${TRY_COMPILE_OUTPUT}")
endif (ARRAY0_WORKS)
