# Distributed under the MIT License.
# See LICENSE.txt for details.

set(ERROR_HANDLING_TESTS
    ErrorHandling/AbortWithErrorMessage.cpp
    ErrorHandling/FloatingPointExceptions.cpp)

set(SPECTRE_TESTS "${SPECTRE_TESTS};${ERROR_HANDLING_TESTS}" PARENT_SCOPE)
