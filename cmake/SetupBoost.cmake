
# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
