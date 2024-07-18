// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is separated from `Test_DataVector.cpp` in an effort to parallelize
// the test builds.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
void test_data_vector_multiple_operand_math() {
  const TestHelpers::VectorImpl::Bound generic{{-10.0, 10.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.1, 10.0}};

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

  const auto test_cross = [](const DataVector& dv1, const DataVector& dv2) {
    return DataVector{{dv1[1] * dv2[2] - dv1[2] * dv2[1],
                       dv1[2] * dv2[0] - dv1[0] * dv2[2],
                       dv1[0] * dv2[1] - dv1[1] * dv2[0]}};
  };

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{generic[0], generic[1]};
  const auto dv1 = make_with_random_values<DataVector>(
      make_not_null(&generator), dist, DataVector{3, 0.0});
  const auto dv2 = make_with_random_values<DataVector>(
      make_not_null(&generator), dist, DataVector{3, 0.0});

  // Check Blaze implementation of cross product
  CHECK(cross(dv1, dv2) == test_cross(dv1, dv2));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.MultipleOperands",
                  "[DataStructures][Unit]") {
  test_data_vector_multiple_operand_math();
}
