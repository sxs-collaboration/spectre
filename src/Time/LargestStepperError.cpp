// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/LargestStepperError.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Time/StepperErrorTolerances.hpp"

double largest_stepper_error(const double values, const double errors,
                             const StepperErrorTolerances& tolerances) {
  return std::abs(errors) /
         (tolerances.absolute +
          tolerances.relative * std::max(abs(values), abs(values + errors)));
}

double largest_stepper_error(const std::complex<double>& values,
                             const std::complex<double>& errors,
                             const StepperErrorTolerances& tolerances) {
  return std::abs(errors) /
         (tolerances.absolute +
          tolerances.relative * std::max(abs(values), abs(values + errors)));
}

double largest_stepper_error(const DataVector& values, const DataVector& errors,
                             const StepperErrorTolerances& tolerances) {
  double result = 0.0;
  for (auto val_it = values.begin(), err_it = errors.begin();
       val_it != values.end(); ++val_it, ++err_it) {
    const double recursive_call_result =
        largest_stepper_error(*val_it, *err_it, tolerances);
    if (recursive_call_result > result) {
      result = recursive_call_result;
    }
  }
  return result;
}

double largest_stepper_error(const ComplexDataVector& values,
                             const ComplexDataVector& errors,
                             const StepperErrorTolerances& tolerances) {
  double result = 0.0;
  for (auto val_it = values.begin(), err_it = errors.begin();
       val_it != values.end(); ++val_it, ++err_it) {
    const double recursive_call_result =
        largest_stepper_error(*val_it, *err_it, tolerances);
    if (recursive_call_result > result) {
      result = recursive_call_result;
    }
  }
  return result;
}
