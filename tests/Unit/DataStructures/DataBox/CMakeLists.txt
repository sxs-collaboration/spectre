# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_DataBox")

set(LIBRARY_SOURCES
  Test_BaseTags.cpp
  Test_DataBox.cpp
  Test_DataBoxDocumentation.cpp
  Test_DataBoxPrefixes.cpp
  Test_DataBoxTag.cpp
  Test_Deferred.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/DataBox"
  "${LIBRARY_SOURCES}"
  "DataStructures;ErrorHandling;Utilities"
  )
