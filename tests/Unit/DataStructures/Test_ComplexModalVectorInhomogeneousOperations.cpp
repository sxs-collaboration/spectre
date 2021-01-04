// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file has been separated from `Test_ComplexModalVector.cpp` in an effort
// to parallelize the test builds.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <tuple>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"   // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

// IWYU pragma: no_include <algorithm>

void test_complex_modal_vector_inhomogeneous_binary_math() noexcept {
  const TestHelpers::VectorImpl::Bound generic{{-10.0, 10.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.1, 10.0}};

  const auto just_element_with_modal_vector_ops =
      std::make_tuple(std::make_tuple(funcl::Multiplies<>{},
                                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, double,
      ComplexModalVector>(just_element_with_modal_vector_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      std::complex<double>, ComplexModalVector>(
      just_element_with_modal_vector_ops);

  const auto just_modal_vector_with_element_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, double>(just_modal_vector_with_element_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, std::complex<double>>(
      just_modal_vector_with_element_ops);

  const auto just_modal_vector_with_modal_vector_ops =
      std::make_tuple(std::make_tuple(funcl::MinusAssign<>{},
                                      std::make_tuple(generic, generic)),
                      std::make_tuple(funcl::PlusAssign<>{},
                                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, ComplexModalVector>(
      just_modal_vector_with_modal_vector_ops);

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly,
      ComplexModalVector, ModalVector>(just_modal_vector_with_modal_vector_ops);
}

SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexModalVector.InhomogeneousOperations",
    "[DataStructures][Unit]") {
  test_complex_modal_vector_inhomogeneous_binary_math();
}
