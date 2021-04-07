// Distributed under the MIT License.
// See LICENSE.txt for details.

// These error tests are in a separate file so `Test_DenseMatrix.cpp` can be
// compiled into `TestArchitectureVectorization`

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DenseMatrix.hpp"
#include "Framework/TestCreation.hpp"

// [[OutputRegex, All matrix columns must have the same size.]]
SPECTRE_TEST_CASE("Unit.DataStructures.DenseMatrix.InvalidOptions",
                  "[DataStructures][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<DenseMatrix<double, blaze::rowMajor>>(
      "[[1], [1, 2], [1, 2]]");
}
