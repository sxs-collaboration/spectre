# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Catch 1.8 REQUIRED)

include_directories(SYSTEM "${CATCH_INCLUDE_DIR}")
message(STATUS "Catch include: ${CATCH_INCLUDE_DIR}")
message(STATUS "Catch version: ${CATCH_VERSION}")
