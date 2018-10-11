# Distributed under the MIT License
# See LICENSE.txt for details

find_package(SSHT REQUIRED)

spectre_include_directories("${SSHT_INCLUDE_DIRS}")
list(APPEND SPECTRE_LIBRARIES ${SSHT_LIBRARIES})

message(STATUS "SSHT library: ${SSHT_LIBRARIES}")
message(STATUS "SSHT include: ${SSHT_INCLUDE_DIRS}")
