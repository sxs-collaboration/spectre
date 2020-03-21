# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Git REQUIRED)

configure_file(
  ${CMAKE_SOURCE_DIR}/src/Informer/InfoAtLink.cpp
  ${CMAKE_BINARY_DIR}/Informer/InfoAtLink.cpp
  )
# Configure info from build to give access to unit test path,
# SpECTRE version, etc. (things known at CMake time)
configure_file(
  ${CMAKE_SOURCE_DIR}/src/Informer/InfoAtCompile.cpp
  ${CMAKE_BINARY_DIR}/Informer/InfoAtCompile.cpp
  )

option(USE_FORMALINE
  "Use Formaline to encode the source tree into executables and output files."
  ON)

# APPLE instead of ${APPLE} is intentional
if (APPLE)
  set(USE_FORMALINE OFF)
endif (APPLE)

if (USE_FORMALINE)
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/Formaline.sh
    ${CMAKE_BINARY_DIR}/tmp/Formaline.sh
    @ONLY
    )
else()
  file(REMOVE ${CMAKE_BINARY_DIR}/tmp/Formaline.sh)
endif()

option(
  STUB_EXECUTABLE_OBJECT_FILES
  "Replace executable object files with stubs to reduce disk usage."
  OFF
  )

set(WRAP_EXECUTABLE_LINKER_USE_STUB_OBJECT_FILES "false")
if (STUB_EXECUTABLE_OBJECT_FILES)
  set(WRAP_EXECUTABLE_LINKER_USE_STUB_OBJECT_FILES "true")
endif (STUB_EXECUTABLE_OBJECT_FILES)

configure_file(
  ${CMAKE_SOURCE_DIR}/tools/WrapExecutableLinker.sh
  ${CMAKE_BINARY_DIR}/tmp/WrapExecutableLinker.sh
  @ONLY
  )

option(
  STUB_LIBRARY_OBJECT_FILES
  "Replace library object files with stubs to reduce disk usage."
  OFF
  )

set(WRAP_LIBRARY_LINKER_USE_STUB_OBJECT_FILES "false")
if (STUB_LIBRARY_OBJECT_FILES)
  set(WRAP_LIBRARY_LINKER_USE_STUB_OBJECT_FILES "true")
endif (STUB_LIBRARY_OBJECT_FILES)

configure_file(
  ${CMAKE_SOURCE_DIR}/tools/WrapLibraryLinker.sh
  ${CMAKE_BINARY_DIR}/tmp/WrapLibraryLinker.sh
  @ONLY
  )
