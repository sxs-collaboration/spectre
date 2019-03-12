// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.Convergence.Reason",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(get_output(Convergence::Reason::MaxIterations) == "MaxIterations");
  CHECK(get_output(Convergence::Reason::AbsoluteResidual) ==
        "AbsoluteResidual");
  CHECK(get_output(Convergence::Reason::RelativeResidual) ==
        "RelativeResidual");
}
