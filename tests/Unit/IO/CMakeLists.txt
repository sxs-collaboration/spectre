# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_IO")

set(LIBRARY_SOURCES
  Test_ComposeTable.cpp
  Test_Connectivity.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  IO
  Informer
  Utilities
  )

add_subdirectory(Exporter)
add_subdirectory(External)
add_subdirectory(H5)
add_subdirectory(Importers)
add_subdirectory(Logging)
add_subdirectory(Observers)
