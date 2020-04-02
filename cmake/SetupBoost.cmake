# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.60.0 REQUIRED COMPONENTS program_options)

message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Boost Version:  ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\n"
  )

# Boost organizes targets as:
# - Boost::boost is the header-only parts of Boost
# - Boost::COMPONENT are the components that need linking, e.g. program_options
