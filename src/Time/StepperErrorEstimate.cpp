// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/StepperErrorEstimate.hpp"

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

StepperErrorEstimate::StepperErrorEstimate(const Time step_time_in,
                                           const TimeDelta step_size_in,
                                           const size_t order_in,
                                           const double step_error_in)
    : step_time(step_time_in), step_size(step_size_in), order(order_in) {
  gsl::at(errors, order).emplace(step_error_in);
}

double StepperErrorEstimate::step_error() const {
  return gsl::at(errors, order).value();
}

void StepperErrorEstimate::pup(PUP::er& p) {
  p | step_time;
  p | step_size;
  p | order;
  p | errors;
}
