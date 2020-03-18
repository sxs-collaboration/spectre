# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Blaze 3.2 REQUIRED)

message(STATUS "Blaze incl: ${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze vers: ${BLAZE_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Blaze Version:  ${BLAZE_VERSION}\n"
  )

add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})
set_property(TARGET Blaze PROPERTY
  INTERFACE_LINK_LIBRARIES Lapack)
