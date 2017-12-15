# Distributed under the MIT License.
# See LICENSE.txt for details.

set(SPECTRE_UNIT_MATH_FUNCTIONS
  PointwiseFunctions/MathFunctions/Test_Gaussian.cpp
  PointwiseFunctions/MathFunctions/Test_PowX.cpp
  PointwiseFunctions/MathFunctions/Test_Sinusoid.cpp
  PointwiseFunctions/MathFunctions/Test_TensorProduct.cpp
  )

set(SPECTRE_UNIT_POINTWISE_FUNCTIONS
  "${SPECTRE_UNIT_POINTWISE_FUNCTIONS};${SPECTRE_UNIT_MATH_FUNCTIONS}"
  PARENT_SCOPE)
