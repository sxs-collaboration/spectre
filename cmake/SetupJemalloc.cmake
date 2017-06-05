# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(JEMALLOC REQUIRED)


include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
set(SPECTRE_LIBRARIES "${SPECTRE_LIBRARIES};${JEMALLOC_LIBRARIES}")

message(STATUS "jemalloc libs: " ${JEMALLOC_LIBRARIES})
message(STATUS "jemalloc incl: " ${JEMALLOC_INCLUDE_DIRS})
