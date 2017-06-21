# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(GSL REQUIRED)

include_directories(SYSTEM ${GSL_INCLUDE_DIR})
list(APPEND SPECTRE_LIBRARIES ${GSL_LIBRARIES})

message(STATUS "GSL libs: ${GSL_LIBRARIES}")
message(STATUS "GSL incl: ${GSL_INCLUDE_DIR}")
message(STATUS "GSL vers: ${GSL_VERSION}")
