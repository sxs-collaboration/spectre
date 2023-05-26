# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AbortWithErrorMessage.cpp
  Breakpoint.cpp
  FloatingPointExceptions.cpp
  SegfaultHandler.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AbortWithErrorMessage.hpp
  Assert.hpp
  Breakpoint.hpp
  Error.hpp
  Exceptions.hpp
  ExpectsAndEnsures.hpp
  FloatingPointExceptions.hpp
  SegfaultHandler.hpp
  StaticAssert.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Charmxx::charmxx
  PUBLIC
  SystemUtilities
  )

# For `backtrace` functionality
if (APPLE)
  target_compile_options(${LIBRARY} INTERFACE "-dynamic")
  target_link_options(${LIBRARY} INTERFACE "-rdynamic")
endif()
