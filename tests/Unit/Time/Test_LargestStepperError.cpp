// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/StepperErrorTolerances.hpp"

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
    const ComplexDataVector values{std::complex<double>{10.0, 3.0},
                                   std::complex<double>{0.0, 0.0}};
    const ComplexDataVector errors{std::complex<double>{5.0, 1.0},
                                   std::complex<double>{1.0, 3.0}};
    const StepperErrorTolerances mostly_abs{.absolute = 2.0, .relative = 0.1};
    const StepperErrorTolerances mostly_rel{.absolute = 0.1, .relative = 2.0};
    CHECK(largest_stepper_error(values, errors, mostly_abs) ==
          largest_stepper_error(values[0], errors[0], mostly_abs));
    CHECK(largest_stepper_error(values, errors, mostly_rel) ==
          largest_stepper_error(values[1], errors[1], mostly_rel));
  }
}
