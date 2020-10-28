// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// These error tests are in a separate file so `Test_DataVector.cpp` can be
/// compiled into `TestArchitectureVectorization`

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <tuple>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.DataVector.ExpressionAssignError",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<DataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::ExpressionAssign);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.RefDiffSize",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<DataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Copy);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.MoveRefDiffSize",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<DataVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Move);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
