# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

set(LIBRARY_SOURCES
    AbortWithErrorMessage.cpp
    FloatingPointExceptions.cpp)

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AbortWithErrorMessage.hpp
  Assert.hpp
  Error.hpp
  Exceptions.hpp
  ExpectsAndEnsures.hpp
  FloatingPointExceptions.hpp
  StaticAssert.hpp
  )
