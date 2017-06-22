# Distributed under the MIT License.
# See LICENSE.txt for details.

set(ERROR_HANDLING_TESTS
    ErrorHandling/Test_AbortWithErrorMessage.cpp
    ErrorHandling/Test_AssertAndError.cpp
    ErrorHandling/Test_FloatingPointExceptions.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${ERROR_HANDLING_TESTS}" PARENT_SCOPE)
