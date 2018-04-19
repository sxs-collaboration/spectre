# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)
spectre_include_directories(${Boost_INCLUDE_DIR})
list(APPEND SPECTRE_LIBRARIES ${Boost_LIBRARIES})
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Boost Version:  ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\n"
  )
