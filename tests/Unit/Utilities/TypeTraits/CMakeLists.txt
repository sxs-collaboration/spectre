# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_TypeTraits")

set(LIBRARY_SOURCES
  Test_FunctionInfo.cpp
  Test_GetFundamentalType.cpp
  Test_IsComplexOfFundamental.cpp
  Test_IsInteger.cpp
  Test_IsStreamable.cpp
  Test_RemoveReferenceWrapper.cpp
  )

add_test_library(
  ${LIBRARY}
  "Utilities/TypeTraits"
  "${LIBRARY_SOURCES}"
  ""
  )

