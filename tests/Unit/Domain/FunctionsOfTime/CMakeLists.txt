# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_FunctionsOfTime")

set(LIBRARY_SOURCES
  Test_FixedSpeedCubic.cpp
  Test_FunctionOfTimeHelpers.cpp
  Test_FunctionsOfTimeAreReady.cpp
  Test_PiecewisePolynomial.cpp
  Test_QuaternionFunctionOfTime.cpp
  Test_QuaternionHelpers.cpp
  Test_ReadSpecPiecewisePolynomial.cpp
  Test_SettleToConstant.cpp
  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/FunctionsOfTime"
  "${LIBRARY_SOURCES}"
  "DataStructures;DomainCreators;FunctionsOfTime;IO;Informer;Options;Utilities"
  )
