# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EXECUTABLE SingletonHelloWorld)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  SingletonHelloWorld.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  Charmxx::main
  DataStructures
  Informer
  Options
  Parallel
  Printf
  SystemUtilities
  Utilities
  )
