# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataBox")

set(LIBRARY_SOURCES
  Test_BaseTags.cpp
  Test_DataBox.cpp
  Test_DataBoxDocumentation.cpp
  Test_DataBoxPrefixes.cpp
  Test_ObservationBox.cpp
  Test_PrefixHelpers.cpp
  Test_Protocols.cpp
  Test_TagName.cpp
  Test_TagTraits.cpp
  Test_TestHelpers.cpp
  Test_ValidateSelection.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  ErrorHandling
  Options
  Utilities
)
