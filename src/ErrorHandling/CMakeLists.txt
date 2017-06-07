# Distributed under the MIT License.
# See LICENSE.txt for details.

set(library ErrorHandling)

set(library_sources
    AbortWithErrorMessage.cpp
    FloatingPointExceptions.cpp)

add_library(${library} ${library_sources})

# ${library} is considered a core library on which all executables depend
set(SPECTRE_CORE_LIBRARIES "${SPECTRE_CORE_LIBRARIES};${library}" PARENT_SCOPE)
