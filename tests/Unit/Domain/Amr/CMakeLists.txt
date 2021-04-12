# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Amr")

set(LIBRARY_SOURCES
  Actions/Test_Initialize.cpp
  Test_Flag.cpp
  Test_Helpers.cpp
  Test_Tags.cpp
  Test_UpdateAmrDecision.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/Amr"
  "${LIBRARY_SOURCES}"
  "Amr;DomainStructure;Utilities"
  )
