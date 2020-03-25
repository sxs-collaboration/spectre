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

# This is needed if your Boost version is newer than your CMake version
# or if you have an old version of CMake (<3.5)
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_property(TARGET Boost::boost PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif(NOT TARGET Boost::boost)

if(NOT TARGET Boost::program_options)
  add_library(Boost::program_options INTERFACE IMPORTED)
  set_property(TARGET Boost::program_options
    APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
  set_property(TARGET Boost::program_options PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIRS})
endif(NOT TARGET Boost::program_options)
