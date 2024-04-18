# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ErrorHandling")

set(LIBRARY_SOURCES
  Test_AbortWithErrorMessage.cpp
  Test_AssertAndError.cpp
  Test_CaptureForError.cpp
  Test_Error.cpp
  Test_Exceptions.cpp
  Test_FloatingPointExceptions.cpp
  Test_SegfaultHandler.cpp
  Test_Strerror.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}"
  # Run in Charm++ environment to avoid issues with the segfault handler test
  # on some systems (macOS).
  WITH_CHARM
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  ErrorHandling
  )
