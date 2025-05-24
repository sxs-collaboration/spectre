// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"

#include "Utilities/ConstantExpressions.hpp"

double ScalarTensor::nonic_ramp_function(const double time,
                                         const double start_time,
                                         const double ramp_time) {
  double ramp_factor = 1.0;

  if (time < start_time) {
    // Before ramping on
    ramp_factor = 0.0;
  } else if (time < ramp_time + start_time) {
    // Ramping on with ramp function given in Eq. B4 of 1911.02588
    const double t = (time - start_time) / ramp_time;
    // Evaluate the polynomial using Horner's method
    ramp_factor =
        pow<5>(t) * (126. + t * (-420. + t * (540. + t * (-315. + 70. * t))));
  }
  // Otherwise the ramp_factor is 1.0 (the default value set above)
  return ramp_factor;
}

double ScalarTensor::nonic_ramp_function(
    const double time, const std::pair<double, double> start_and_ramp_times) {
  const double start_time = start_and_ramp_times.first;
  const double ramp_time = start_and_ramp_times.second;
  return ScalarTensor::nonic_ramp_function(time, start_time, ramp_time);
}
