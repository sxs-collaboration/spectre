# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(Git REQUIRED)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/Informer/InfoFromBuild.cpp
    ${CMAKE_BINARY_DIR}/Informer/InfoFromBuild.cpp
)

configure_file(
    ${CMAKE_SOURCE_DIR}/tools/WrapLinker.sh
    ${CMAKE_BINARY_DIR}/WrapLinker.sh
    @ONLY
)

string(
    REGEX REPLACE "<CMAKE_CXX_COMPILER>"
    "${CMAKE_BINARY_DIR}/WrapLinker.sh <CMAKE_CXX_COMPILER>"
    CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}"
)
