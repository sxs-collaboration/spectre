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

# We always want a detailed backtrace of template errors to make debugging them
# easier
set(CMAKE_CXX_FLAGS "-ftemplate-backtrace-limit=0 ${CMAKE_CXX_FLAGS}")

# We disable thread safety of Boost::shared_ptr since it makes them faster
# to use and we do not share them between threads. If a thread-safe
# shared_ptr is desired it must be implemented to work with Charm++'s threads
# anyway.
add_definitions(-DBOOST_SP_DISABLE_THREADS)
