# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Git REQUIRED)

configure_file(
  ${CMAKE_SOURCE_DIR}/src/Informer/InfoFromBuild.cpp
  ${CMAKE_BINARY_DIR}/Informer/InfoFromBuild.cpp
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
