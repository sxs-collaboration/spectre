# Distributed under the MIT License.
# See LICENSE.txt for details.

list(APPEND UTILITIES_TESTS Utilities/ConstantExpressions.cpp)

# CMake cannot append to a list in the parent scope
list(APPEND UTILITIES_TESTS ${SPECTRE_TESTS})
set(SPECTRE_TESTS "${UTILITIES_TESTS}" PARENT_SCOPE)

