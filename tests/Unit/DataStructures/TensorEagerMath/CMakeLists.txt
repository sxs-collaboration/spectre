# Distributed under the MIT License.
# See LICENSE.txt for details.

set(TENSOR_EAGER_MATH_TESTS
    DataStructures/TensorEagerMath/Test_Determinant.cpp
    DataStructures/TensorEagerMath/Test_DeterminantAndInverse.cpp
    DataStructures/TensorEagerMath/Test_DivideBy.cpp
    DataStructures/TensorEagerMath/Test_Magnitude.cpp
    )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${TENSOR_EAGER_MATH_TESTS}" PARENT_SCOPE)
