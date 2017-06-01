# Distributed under the MIT License.
# See LICENSE.txt for details.

list(APPEND ERROR_HANDLING_TESTS ErrorHandling/FloatingPointExceptions.cpp)

# CMake cannot append to a list in the parent scope
list(APPEND ERROR_HANDLING_TESTS ${SPECTRE_TESTS})
set(SPECTRE_TESTS "${ERROR_HANDLING_TESTS}" PARENT_SCOPE)

