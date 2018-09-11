# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

set(LIBRARY_SOURCES
    AbortWithErrorMessage.cpp
    FloatingPointExceptions.cpp)

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})
