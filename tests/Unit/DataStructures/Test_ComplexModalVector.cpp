// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <complex>
#include <tuple>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "ErrorHandling/Error.hpp"           // IWYU pragma: keep
#include "Utilities/DereferenceWrapper.hpp"  // IWYU pragma: keep
#include "Utilities/Functional.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep
#include "tests/Unit/DataStructures/VectorImplTestHelper.hpp"

// IWYU pragma: no_include <algorithm>

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexModalVector.ExpressionAssignError",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::ExpressionAssign);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
        "Unit.DataStructures.ComplexModalVector.RefDiffSize",
        "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Copy);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.ComplexModalVector.MoveRefDiffSize",
    "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  TestHelpers::VectorImpl::vector_ref_test_size_error<ComplexModalVector>(
      TestHelpers::VectorImpl::RefSizeErrorTestKind::Move);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

void test_complex_modal_vector_math() noexcept {
  const TestHelpers::VectorImpl::Bound generic{{-100.0, 100.0}};
  const TestHelpers::VectorImpl::Bound positive{{0.01, 100.0}};

  const auto unary_ops = std::make_tuple(
      std::make_tuple(funcl::Conj<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ComplexModalVector>(unary_ops);

  const auto real_unary_ops = std::make_tuple(
      std::make_tuple(funcl::Imag<>{}, std::make_tuple(generic)),
      std::make_tuple(funcl::Real<>{}, std::make_tuple(generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Normal, ModalVector>(real_unary_ops);

  const auto binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexModalVector,
      ModalVector>(binary_ops);

  const auto cascaded_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<funcl::Plus<>, funcl::Identity>{},
                      std::make_tuple(generic, generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict, ComplexModalVector,
      ModalVector>(cascaded_ops);

  const auto array_binary_ops = std::make_tuple(
      std::make_tuple(funcl::Minus<>{}, std::make_tuple(generic, generic)),
      std::make_tuple(funcl::Plus<>{}, std::make_tuple(generic, generic)));

  TestHelpers::VectorImpl::test_functions_with_vector_arguments<
      TestHelpers::VectorImpl::TestKind::Strict,
      std::array<ComplexModalVector, 2>>(array_binary_ops);

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

SPECTRE_TEST_CASE("Unit.DataStructures.ComplexModalVector",
                  "[DataStructures][Unit]") {
  {
    INFO("test construct and assign");
    TestHelpers::VectorImpl::vector_test_construct_and_assign<
        ComplexModalVector, std::complex<double>>();
  }
  {
    INFO("test serialize and deserialize");
    TestHelpers::VectorImpl::vector_test_serialize<ComplexModalVector,
                                                   std::complex<double>>();
  }
  {
    INFO("test set_data_ref functionality");
    TestHelpers::VectorImpl::vector_test_ref<ComplexModalVector,
                                             std::complex<double>>();
  }
  {
    INFO("test math after move");
    TestHelpers::VectorImpl::vector_test_math_after_move<
        ComplexModalVector, std::complex<double>>();
  }
  {
    INFO("test ComplexModalVector math operations");
    test_complex_modal_vector_math();
  }
}
