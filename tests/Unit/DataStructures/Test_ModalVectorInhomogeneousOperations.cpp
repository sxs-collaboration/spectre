// Distributed under the MIT License.
// See LICENSE.txt for details.

// This file is separated from `Test_ModalVector.cpp` in an effort to
// parallelize the test builds.

#include "Framework/TestingFramework.hpp"

#include <tuple>

#include "DataStructures/ModalVector.hpp"
#include "Helpers/DataStructures/VectorImplTestHelper.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
void test_modal_vector_inhomogeneous_binary_math() {
  const TestHelpers::VectorImpl::Bound generic{{-10.0, 10.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.1, 10.0}};

  const auto just_double_with_modal_vector_ops =
      std::make_tuple(std::make_tuple(funcl::Multiplies<>{},
                                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, double,
      ModalVector>(just_double_with_modal_vector_ops);

  const auto just_modal_vector_with_double_ops = std::make_tuple(
      std::make_tuple(funcl::Multiplies<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Divides<>{}, std::make_tuple(generic, positive)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, ModalVector,
      double>(just_modal_vector_with_double_ops);

  const auto just_modal_vector_with_modal_vector_ops =
      std::make_tuple(std::make_tuple(funcl::MinusAssign<>{},
                                      std::make_tuple(generic, generic)),
                      std::make_tuple(funcl::PlusAssign<>{},
                                      std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::GivenOrderOfArgumentsOnly, ModalVector,
      ModalVector>(just_modal_vector_with_modal_vector_ops);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.ModalVector.InhomogeneousOperations",
                  "[DataStructures][Unit]") {
  test_modal_vector_inhomogeneous_binary_math();
}
