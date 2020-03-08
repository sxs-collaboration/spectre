# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_TypeTraits")

set(LIBRARY_SOURCES
  Test_FunctionInfo.cpp
  Test_IsComplexOfFundamental.cpp
  )

add_test_library(
  ${LIBRARY}
  "Utilities/TypeTraits"
  "${LIBRARY_SOURCES}"
  ""
  )

