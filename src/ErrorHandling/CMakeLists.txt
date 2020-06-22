# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AbortWithErrorMessage.cpp
  FloatingPointExceptions.cpp
  )

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
