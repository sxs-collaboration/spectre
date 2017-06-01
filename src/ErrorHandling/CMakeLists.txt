# Distributed under the MIT License.
# See LICENSE.txt for details.

set(library ErrorHandling)

set(library_sources
    FloatingPointExceptions.cpp
    AbortWithErrorMessage.cpp)

add_library(${library} ${library_sources})
