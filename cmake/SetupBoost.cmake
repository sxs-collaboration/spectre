# Distributed under the MIT License.
# See LICENSE.txt for details.

option(
  BUILD_PYTHON_BINDINGS
  "Build the python bindings for SpECTRE"
  OFF
  )

if("" STREQUAL "${PYTHONLIBS_VERSION_STRING}")
  message(FATAL_ERROR
    "Must find python libraries before finding the Boost libraries because "
    "we need the Boost libraries to be built against the same version of "
    "the python libraries.")
endif()

string(REPLACE "." ";" SPECTRE_PYTHON_VERS_LIST ${PYTHONLIBS_VERSION_STRING})
list(GET SPECTRE_PYTHON_VERS_LIST 0 PYTHON_LIBS_MAJOR_VERSION)
list(GET SPECTRE_PYTHON_VERS_LIST 1 PYTHON_LIBS_MINOR_VERSION)
list(GET SPECTRE_PYTHON_VERS_LIST 2 PYTHON_LIBS_PATCH_VERSION)

# Do first find to get version.
find_package(Boost 1.60.0 REQUIRED)

if(BUILD_PYTHON_BINDINGS)
  set(SPECTRE_BOOST_PYTHON_COMPONENT "python")

  if ("${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}"
      VERSION_GREATER 1.66.0)
    set(SPECTRE_BOOST_PYTHON_COMPONENT
      "python${PYTHON_LIBS_MAJOR_VERSION}${PYTHON_LIBS_MINOR_VERSION}")
  endif()
endif()

find_package(
  Boost
  1.60.0
  REQUIRED
  COMPONENTS
  program_options
  ${SPECTRE_BOOST_PYTHON_COMPONENT})
spectre_include_directories(${Boost_INCLUDE_DIR})
list(APPEND SPECTRE_LIBRARIES ${Boost_LIBRARIES})
message(STATUS "Boost include: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "Boost Version:  ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}\n"
  )
