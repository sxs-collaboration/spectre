# Distributed under the MIT License
# See LICENSE.txt for details

find_package(Libsharp REQUIRED)

spectre_include_directories("${LIBSHARP_INCLUDE_DIRS}")
list(APPEND SPECTRE_LIBRARIES ${LIBSHARP_LIBRARIES})
list(APPEND SPECTRE_LIBRARIES ${LIBSHARP_LIBFFTPACK})
list(APPEND SPECTRE_LIBRARIES ${LIBSHARP_LIBCUTILS})

message(STATUS "libsharp library: ${LIBSHARP_LIBRARIES}")
message(STATUS "libsharp library: ${LIBSHARP_LIBFFTPACK}")
message(STATUS "libsharp library: ${LIBSHARP_LIBCUTILS}")
message(STATUS "libsharp include: ${LIBSHARP_INCLUDE_DIRS}")
