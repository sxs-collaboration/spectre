# Distributed under the MIT License.
# See LICENSE.txt for details.

option(MACHINE "Select a machine that we know how to run on, such as a \
particular supercomputer" OFF)

if(NOT MACHINE)
  set(MACHINE "UNKNOWN")
endif()

message(STATUS "Selected machine: ${MACHINE}")
