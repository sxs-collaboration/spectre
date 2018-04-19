# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(GSL REQUIRED)

spectre_include_directories(${GSL_INCLUDE_DIR})
list(APPEND SPECTRE_LIBRARIES ${GSL_LIBRARIES})
# Extract the path where the shared libraries are and point the linker to
# that directory
list(GET GSL_LIBRARIES 0 FIRST_GSL_LIBRARY)
string(REGEX REPLACE "libgsl(cblas)?.(a|so|dylib)" ""
    GSL_LIBRARY_PATH ${FIRST_GSL_LIBRARY})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${GSL_LIBRARY_PATH}")
list(APPEND SPECTRE_LIBRARIES "-lgsl")
list(APPEND SPECTRE_LIBRARIES "-lgslcblas")

message(STATUS "GSL libs: ${GSL_LIBRARIES}")
message(STATUS "GSL incl: ${GSL_INCLUDE_DIR}")
message(STATUS "GSL vers: ${GSL_VERSION}")

file(APPEND
  "${CMAKE_BINARY_DIR}/LibraryVersions.txt"
  "GSL Version:  ${GSL_VERSION}\n"
  )
