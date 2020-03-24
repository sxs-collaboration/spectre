// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <tuple>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"        // IWYU pragma: keep
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"        // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

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

void test_data_vector_unary_math() noexcept {
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

  // Note that the binary operations have been moved to
  // `Test_DataVectorBinaryOperations.cpp` in an effort to better parallelize
  // the build.
}

namespace {
void test_norms() noexcept {
  // Test l1Norm and l2Norm:
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> dist{-5, 10};
  DataVector vector(30);
  fill_with_random_values(make_not_null(&vector), make_not_null(&gen),
                          make_not_null(&dist));
  double l1norm = 0.0, l2norm = 0.0;
  for (const double value : vector) {
    l1norm += std::abs(value);
    l2norm += square(value);
  }
  l2norm = std::sqrt(l2norm);
  // Since l1Norm(vector) and l2Norm(vector) use SIMD we shouldn't expect the
  // results to be bitwise identical.
  CHECK(l1norm == approx(l1Norm(vector)));
  CHECK(l2norm == approx(l2Norm(vector)));
}
}  // namespace

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
    test_data_vector_unary_math();
  }
  {
    INFO("test norms of DataVectors");
    test_norms();
  }
}
