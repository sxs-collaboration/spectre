# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_MathFunctions")

set(LIBRARY_SOURCES
  Test_Gaussian.cpp
  Test_PowX.cpp
  Test_Sinusoid.cpp
  Test_TensorProduct.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/MathFunctions/"
  "${LIBRARY_SOURCES}"
  "MathFunctions;Utilities"
  )
