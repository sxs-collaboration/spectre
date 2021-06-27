# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Arpack REQUIRED)

add_library(Arpack INTERFACE IMPORTED)

if(ARPACK_FOUND)
  message(STATUS "Arpack libs: " ${ARPACK_LIBRARIES})
else()
  message(FATAL_ERROR "Arpack not found\n")
endif()

set_property(TARGET Arpack
  APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${ARPACK_LIBRARIES})
set_property(TARGET Arpack
  APPEND PROPERTY INTERFACE_LINK_OPTIONS ${ARPACK_LINKER_FLAGS})

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Arpack
  )
