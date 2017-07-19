# Distributed under the MIT License.
# See LICENSE.txt for details.

set(
    CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -DSPECTRE_DEBUG"
)

# Always build with -g so we can view backtraces, etc. when production code
# fails. This can be overridden by passing `-D CMAKE_CXX_FLAGS="-g0"` to CMake
set(CMAKE_CXX_FLAGS "-g ${CMAKE_CXX_FLAGS}")

# Always compile only for the current architecture. This can be overridden
# by passing `-D CMAKE_CXX_FLAGS="-march=THE_ARCHITECTURE"` to CMake
set(CMAKE_CXX_FLAGS "-march=native ${CMAKE_CXX_FLAGS}")
