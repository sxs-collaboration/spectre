# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Framework)

add_subdirectory(Tests)

set(LIBRARY_SOURCES
  SetupLocalPythonEnvironment.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE
  Boost::boost
  DataStructures
  DataStructuresHelpers
  Options
  Parallel
  Utilities

  PRIVATE
  Informer
  NumPy

  PUBLIC
  ErrorHandling
  )

target_include_directories(
  ${LIBRARY}
  SYSTEM
  PUBLIC ${NUMPY_INCLUDE_DIRS}
  )
