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

# APPLE instead of ${APPLE} is intentional
if (NOT APPLE)
  configure_file(
    ${CMAKE_SOURCE_DIR}/tools/Formaline.sh
    ${CMAKE_BINARY_DIR}/tmp/Formaline.sh
    @ONLY
    )
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/tools/WrapLinker.sh
  ${CMAKE_BINARY_DIR}/tmp/WrapLinker.sh
  @ONLY
  )

string(
  REGEX REPLACE "<CMAKE_CXX_COMPILER>"
  "${CMAKE_BINARY_DIR}/tmp/WrapLinker.sh <CMAKE_CXX_COMPILER>"
  CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}"
  )
