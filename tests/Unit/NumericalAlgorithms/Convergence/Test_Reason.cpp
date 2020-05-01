// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ConvergenceReasonOptionTag {
  using type = Convergence::Reason;
  static std::string name() noexcept { return "ConvergenceReason"; }
  static constexpr OptionString help{"A convergence reason for testing"};
};

template <Convergence::Reason TheConvergenceReason>
void test_construct_from_options() noexcept {
  Options<tmpl::list<ConvergenceReasonOptionTag>> opts("");
  opts.parse("ConvergenceReason: " + get_output(TheConvergenceReason) + "\n");
  CHECK(opts.get<ConvergenceReasonOptionTag>() == TheConvergenceReason);
}

void test_construct_from_options_fail() noexcept {
  Options<tmpl::list<ConvergenceReasonOptionTag>> opts("");
  opts.parse("ConvergenceReason: Miracle\n");  // Meant to fail.
  opts.get<ConvergenceReasonOptionTag>();
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Reason",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(get_output(Convergence::Reason::MaxIterations) == "MaxIterations");
  CHECK(get_output(Convergence::Reason::AbsoluteResidual) ==
        "AbsoluteResidual");
  CHECK(get_output(Convergence::Reason::RelativeResidual) ==
        "RelativeResidual");

  test_construct_from_options<Convergence::Reason::MaxIterations>();
  test_construct_from_options<Convergence::Reason::AbsoluteResidual>();
  test_construct_from_options<Convergence::Reason::RelativeResidual>();
}

// [[OutputRegex, Failed to convert "Miracle" to Convergence::Reason]]
SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Reason.FailOptionParsing",
                  "[Unit][NumericalAlgorithms]") {
  ERROR_TEST();
  test_construct_from_options_fail();
}
