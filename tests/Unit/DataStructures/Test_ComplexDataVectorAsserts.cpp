// Distributed under the MIT License.
// See LICENSE.txt for details.

// These error tests are in a separate file so `Test_ComplexDataVector.cpp` can
// be compiled into `TestArchitectureVectorization`

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <tuple>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// IWYU pragma: no_include "DataStructures/DenseMatrix.hpp"

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexDataVector.ExpressionAssignError",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexDataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::ExpressionAssign);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexDataVector.RefDiffSize",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexDataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Copy);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexDataVector.MoveRefDiffSize",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexDataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Move);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
