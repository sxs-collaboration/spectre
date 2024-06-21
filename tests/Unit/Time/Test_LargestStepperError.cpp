// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Time.LargestStepperError", "[Unit][Time]") {
  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 2.0, .relative = 0.0}) ==
        5.0);
  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 0.0, .relative = 2.0}) ==
        1.0 / 6.0);
  CHECK(largest_stepper_error(0.0, 10.0, {.absolute = 0.0, .relative = 2.0}) ==
        0.5);

  CHECK(largest_stepper_error(20.0, 10.0, {.absolute = 2.0, .relative = 3.0}) ==
        10.0 / (2.0 + 3.0 * 30.0));

  CHECK(largest_stepper_error(std::complex<double>(12.0, 5.0),
                              std::complex<double>(-3.0, -4.0),
                              {.absolute = 2.0, .relative = 3.0}) ==
        approx(5.0 / (2.0 + 3.0 * 13.0)));

  {
    const DataVector values{10.0, 0.0};
    const DataVector errors{5.0, 2.0};
    const StepperErrorTolerances mostly_abs{.absolute = 2.0, .relative = 0.1};
    const StepperErrorTolerances mostly_rel{.absolute = 0.1, .relative = 2.0};
    CHECK(largest_stepper_error(values, errors, mostly_abs) ==
          largest_stepper_error(10.0, 5.0, mostly_abs));
    CHECK(largest_stepper_error(values, errors, mostly_rel) ==
          largest_stepper_error(0.0, 2.0, mostly_rel));
  }

  {
    using Vars = Variables<
        tmpl::list<TestHelpers::Tags::Scalar<>, TestHelpers::Tags::Vector<>>>;
    Vars values(5, 0.0);
    get<1>(get<TestHelpers::Tags::Vector<>>(values))[1] = 3.0;
    Vars errors(5, 0.0);
    get<1>(get<TestHelpers::Tags::Vector<>>(errors))[1] = 6.0;
    const StepperErrorTolerances tolerances{.absolute = 2.0, .relative = 5.0};
    CHECK(largest_stepper_error(values, errors, tolerances) ==
          largest_stepper_error(3.0, 6.0, tolerances));
  }

  {
    using Weighted = SpinWeighted<std::complex<double>, 2>;
    const std::complex<double> value(3.0, 5.0);
    const std::complex<double> error(2.0, 1.0);
    const StepperErrorTolerances tolerances{.absolute = 2.0, .relative = 5.0};
    CHECK(largest_stepper_error(Weighted(value), Weighted(error), tolerances) ==
          largest_stepper_error(value, error, tolerances));
  }
}
