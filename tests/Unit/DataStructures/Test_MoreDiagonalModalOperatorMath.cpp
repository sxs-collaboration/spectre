// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is separated from `Test_DiagonalModalOperator.cpp` in an effort to
// parallelize the test builds.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "DataStructures/DiagonalModalOperator.hpp"
#include "DataStructures/ModalVector.hpp"  // IWYU pragma: keep
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <algorithm>

void test_additional_diagonal_modal_operator_math() {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};

  const auto acting_on_modal_vector = std::make_tuple(std::make_tuple(
      funcl::Multiplies<>{}, std::make_tuple(generic, generic)));

  // the operation isn't really "inplace", but we carefully forbid the operation
  // between two ModalVectors, which will be avoided in the inplace test case,
  // which checks only combinations with the DiagonalModalOperator as the first
  // argument.
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Inplace, DiagonalModalOperator,
      ModalVector>(acting_on_modal_vector);
  // testing the other ordering
  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, ModalVector,
      DiagonalModalOperator>(acting_on_modal_vector);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)),
      std::make_tuple(funcl::Minus<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, DiagonalModalOperator>(
      cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict,
      std::array<DiagonalModalOperator, 2>>(array_binary_ops);
}

SPECTRE_TEST_CASE("Unit.DataStructures.DiagonalModalOperator.AdditionalMath",
                  "[DataStructures][Unit]") {
  test_additional_diagonal_modal_operator_math();
}
