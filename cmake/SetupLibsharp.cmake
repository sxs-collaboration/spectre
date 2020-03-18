# Distributed under the MIT License
# See LICENSE.txt for details

find_package(Libsharp REQUIRED)

message(STATUS "libsharp libs: ${LIBSHARP_LIBRARIES}")
message(STATUS "libsharp libs: ${LIBSHARP_LIBFFTPACK}")
message(STATUS "libsharp libs: ${LIBSHARP_LIBCUTILS}")
message(STATUS "libsharp incl: ${LIBSHARP_INCLUDE_DIRS}")

add_library(Libsharp INTERFACE IMPORTED)
set_property(TARGET Libsharp PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${LIBSHARP_INCLUDE_DIRS})
set_property(TARGET Libsharp PROPERTY
  INTERFACE_LINK_LIBRARIES
  ${LIBSHARP_LIBRARIES}
  ${LIBSHARP_LIBFFTPACK}
  ${LIBSHARP_LIBCUTILS}
  )
