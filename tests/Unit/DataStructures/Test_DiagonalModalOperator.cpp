// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <tuple>

#include "DataStructures/DiagonalModalOperator.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
void test_diagonal_modal_operator_math() {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.01, 100.0}};

  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, DiagonalModalOperator>(
      binary_ops);

  const auto inplace_binary_ops = std::make_tuple(
      std::make_tuple(funcl::DivAssign<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::MinusAssign<>{},
                      std::make_tuple(generic, generic)),
      std::make_tuple(funcl::MultAssign<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::PlusAssign<>{},
                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Inplace, DiagonalModalOperator>(
      inplace_binary_ops);

  // Note that a collection of additional operations that involve acting on
  // modal vectors with diagonal modal operators have been moved to
  // `Test_MoreDiagonalModalOperatorMath.cpp` in an effort to better
  // parallelize the build.
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DiagonalModalOperator",
                  "[DataStructures][Unit]") {
  {
    INFO("test construct and assign");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<
        DiagonalModalOperator, double>();
  }
  {
    INFO("test serialize and deserialize");
    TestHelpers::VectorImpl::vector_test_serialize<DiagonalModalOperator,
                                                   double>();
  }
  {
    INFO("test set_data_ref functionality");
    TestHelpers::VectorImpl::vector_test_ref<DiagonalModalOperator, double>();
  }
  {
    INFO("test math after move");
    TestHelpers::VectorImpl::vector_test_math_after_move<DiagonalModalOperator,
                                                         double>();
  }
  {
    INFO("test DiagonalModalOperator math operations");
    test_diagonal_modal_operator_math();
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<
          DiagonalModalOperator>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::ExpressionAssign),
      Catch::Matchers::ContainsSubstring("Must assign into same size"));
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<
          DiagonalModalOperator>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::Copy),
      Catch::Matchers::ContainsSubstring("Must copy into same size"));
  CHECK_THROWS_WITH(
      TestHelpers::VectorImpl::vector_ref_test_size_error<
          DiagonalModalOperator>(
          TestHelpers::VectorImpl::RefSizeErrorTestKind::Move),
      Catch::Matchers::ContainsSubstring("Must copy into same size"));
#endif
}
