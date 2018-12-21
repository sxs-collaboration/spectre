// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"        // IWYU pragma: keep
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"

// IWYU pragma: no_include <algorithm>

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

void test_data_vector_math() noexcept {
  /// [test_functions_with_vector_arguments_example]
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};
  const TestHelpers::VectorImpl::Bound mone_one{{-1.0, 1.0}};
  const TestHelpers::VectorImpl::Bound gt_one{{1.0, 100.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.01, 100.0}};
  const auto unary_ops = std::make_tuple(
      std::make_tuple(funcl::Abs<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Acos<>{}, std::make_tuple(mone_one)),
      std::make_tuple(funcl::Acosh<>{}, std::make_tuple(gt_one)),
      std::make_tuple(funcl::Asin<>{}, std::make_tuple(mone_one)),
      std::make_tuple(funcl::Asinh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Atan<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Atanh<>{}, std::make_tuple(mone_one)),
      std::make_tuple(funcl::Cbrt<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Cos<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Cosh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Erf<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Exp<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Exp2<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Fabs<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::InvCbrt<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::InvSqrt<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Log<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Log10<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Log2<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Sin<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Sinh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::StepFunction<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Square<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Sqrt<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Tan<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Tanh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<1>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<-2>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<3>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, DataVector>(unary_ops);
  /// [test_functions_with_vector_arguments_example]

  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, DataVector>(binary_ops);

  const auto inplace_binary_ops = std::make_tuple(
      std::make_tuple(funcl::DivAssign<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::MinusAssign<>{},
                      std::make_tuple(generic, generic)),
      std::make_tuple(funcl::MultAssign<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::PlusAssign<>{},
                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Inplace, DataVector>(
      inplace_binary_ops);

  const auto strict_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Atan2<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::Hypot<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, DataVector>(strict_binary_ops);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)),
      std::make_tuple(funcl::Multiplies<funcl::Exp<>, funcl::Sin<>>{},
                      std::make_tuple(generic, generic)),
      std::make_tuple(
          funcl::Plus<funcl::Atan<funcl::Tan<>>, funcl::Divides<>>{},
          std::make_tuple(generic, generic, positive)),
      // step function with multiplication should be tested due to resolved
      // [issue 1122](https://github.com/sxs-collaboration/spectre/issues/1122)
      std::make_tuple(funcl::StepFunction<
                          funcl::Plus<funcl::Multiplies<>, funcl::Identity>>{},
                      std::make_tuple(generic, generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, DataVector>(cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, std::array<DataVector, 2>>(
      array_binary_ops);
}

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector", "[DataStructures][Unit]") {
  {
    INFO("test construct and assign");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<DataVector,
                                                              double>();
  }
  {
    INFO("test serialize and deserialize");
    TestHelpers::VectorImpl::vector_test_serialize<DataVector, double>();
  }
  {
    INFO("test set_data_ref functionality");
    TestHelpers::VectorImpl::vector_test_ref<DataVector, double>();
  }
  {
    INFO("test math after move");
    TestHelpers::VectorImpl::vector_test_math_after_move<DataVector, double>();
  }
  {
    INFO("test DataVector math operations");
    test_data_vector_math();
  }
}
