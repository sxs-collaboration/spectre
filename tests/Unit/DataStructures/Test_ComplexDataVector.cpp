// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <tuple>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Error.hpp"           // IWYU pragma: keep
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"

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

void test_complex_data_vector_math() noexcept {
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
      std::make_tuple(funcl::Conj<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Cos<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Cosh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Exp<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::InvSqrt<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Log<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Sin<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Sinh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Square<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Sqrt<>{}, std::make_tuple(positive)),
      std::make_tuple(funcl::Tan<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Tanh<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<1>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<-2>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::UnaryPow<3>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexDataVector>(unary_ops);

  /// [test_vector_math_usage]
  const auto real_unary_ops = std::make_tuple(
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, DataVector>(real_unary_ops);
  /// [test_vector_math_usage]
  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexDataVector,
      ComplexDataVector>(binary_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, DataVector, ComplexDataVector>(
      binary_ops);

  const auto inplace_binary_ops = std::make_tuple(
      std::make_tuple(funcl::DivAssign<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::MinusAssign<>{},
                      std::make_tuple(generic, generic)),
      std::make_tuple(funcl::MultAssign<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::PlusAssign<>{},
                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Inplace, ComplexDataVector,
      ComplexDataVector>(inplace_binary_ops);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)),
      std::make_tuple(funcl::Multiplies<funcl::Exp<>, funcl::Sin<>>{},
                      std::make_tuple(generic, generic, generic)),
      std::make_tuple(
          funcl::Plus<funcl::Atan<funcl::Tan<>>, funcl::Divides<>>{},
          std::make_tuple(generic, generic, positive)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexDataVector,
      ComplexDataVector, ComplexDataVector>(cascaded_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexDataVector, DataVector,
      ComplexDataVector>(cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict,
      std::array<ComplexDataVector, 2>, std::array<ComplexDataVector, 2>>(
      array_binary_ops);
}

SPECTRE_TEST_CASE("Unit.DataStructures.ComplexDataVector",
                  "[DataStructures][Unit]") {
  {
    INFO("test construct and assign");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<
        ComplexDataVector, std::complex<double>>();
  }
  {
    INFO("test construct and assign from doubles");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<ComplexDataVector,
                                                              double>();
  }
  {
    INFO("test serialize and deserialize");
    TestHelpers::VectorImpl::vector_test_serialize<ComplexDataVector,
                                                   std::complex<double>>();
  }
  {
    INFO("test serialize and deserialize from doubles");
    TestHelpers::VectorImpl::vector_test_serialize<ComplexDataVector, double>();
  }
  {
    INFO("test set_data_ref functionality");
    TestHelpers::VectorImpl::vector_test_ref<ComplexDataVector,
                                             std::complex<double>>();
  }
  {
    INFO("test math after move");
    TestHelpers::VectorImpl::vector_test_math_after_move<
        ComplexDataVector, std::complex<double>>();
  }
  {
    INFO("test math after move using doubles to initialize")
    TestHelpers::VectorImpl::vector_test_math_after_move<ComplexDataVector,
                                                         double>();
  }
  {
    INFO("test ComplexDataVector math operations");
    test_complex_data_vector_math();
  }
}
