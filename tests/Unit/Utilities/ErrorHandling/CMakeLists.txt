# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ErrorHandling")

set(LIBRARY_SOURCES
  Test_AbortWithErrorMessage.cpp
  Test_AssertAndError.cpp
  Test_CanSafelyThrow.cpp
  Test_Exceptions.cpp
  Test_FloatingPointExceptions.cpp
  )

add_test_library(
  ${LIBRARY}
  "Utilities/ErrorHandling"
  "${LIBRARY_SOURCES}"
  "ErrorHandling"
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Framework
  )
