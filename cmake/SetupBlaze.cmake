# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Blaze 3.7 REQUIRED)

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

add_interface_lib_headers(
  TARGET Blaze
  HEADERS
  blaze/math/CustomVector.h
  blaze/math/DynamicMatrix.h
  blaze/math/DynamicVector.h
  blaze/system/Optimizations.h
  blaze/system/Version.h
  blaze/util/typetraits/RemoveConst.h
  )

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Blaze
  )
