// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <Convergence::Reason TheConvergenceReason>
void test_construct_from_options() noexcept {
  const auto created = TestHelpers::test_creation<Convergence::Reason>(
      get_output(TheConvergenceReason));
  CHECK(created == TheConvergenceReason);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Reason",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(get_output(Convergence::Reason::NumIterations) == "NumIterations");
  CHECK(get_output(Convergence::Reason::MaxIterations) == "MaxIterations");
  CHECK(get_output(Convergence::Reason::AbsoluteResidual) ==
        "AbsoluteResidual");
  CHECK(get_output(Convergence::Reason::RelativeResidual) ==
        "RelativeResidual");

  test_construct_from_options<Convergence::Reason::NumIterations>();
  test_construct_from_options<Convergence::Reason::MaxIterations>();
  test_construct_from_options<Convergence::Reason::AbsoluteResidual>();
  test_construct_from_options<Convergence::Reason::RelativeResidual>();
}

// [[OutputRegex, Failed to convert "Miracle" to Convergence::Reason]]
SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Reason.FailOptionParsing",
                  "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  TestHelpers::test_creation<Convergence::Reason>("Miracle");
}
