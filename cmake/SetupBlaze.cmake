# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Blaze REQUIRED)

include_directories(SYSTEM "${BLAZE_INCLUDE_DIR}")
message(STATUS "Blaze include: ${BLAZE_INCLUDE_DIR}")
