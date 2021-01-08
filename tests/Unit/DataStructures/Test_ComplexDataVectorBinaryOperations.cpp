// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is separated from `Test_ComplexDataVector.cpp` in an effort to
// parallelize the test builds.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"   // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include "DataStructures/DenseMatrix.hpp"

void test_complex_data_vector_multiple_operand_math() noexcept {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.01, 100.0}};

  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)),
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexDataVector,
      ComplexDataVector>(binary_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, DataVector, ComplexDataVector>(
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

SPECTRE_TEST_CASE("Unit.DataStructures.ComplexDataVector.MultipleOperands",
                  "[DataStructures][Unit]") {
    test_complex_data_vector_multiple_operand_math();
}
