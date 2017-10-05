# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND SPECTRE_LIBRARIES ${Boost_LIBRARIES})
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
