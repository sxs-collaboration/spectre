# Distributed under the MIT License.
# See LICENSE.txt for details.

# Configure info from build to give access to unit test path,
# SpECTRE version, etc. (things known at CMake time)
configure_file(
  ${CMAKE_SOURCE_DIR}/src/Informer/InfoAtCompile.cpp
  ${CMAKE_BINARY_DIR}/Informer/InfoAtCompile.cpp
  )

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
