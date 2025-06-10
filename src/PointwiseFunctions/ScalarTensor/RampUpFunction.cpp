// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/ScalarTensor/RampUpFunction.hpp"

#include "Utilities/Math.hpp"

double ScalarTensor::nonic_ramp_function(const double time,
                                         const double start_time,
                                         const double ramp_time) {
  // Ramping on with ramp function given in Eq. B4 of 1911.02588
  return smoothstep<4>(start_time, start_time + ramp_time, time);
}

double ScalarTensor::nonic_ramp_function(
    const double time, const std::pair<double, double> start_and_ramp_times) {
  const double start_time = start_and_ramp_times.first;
  const double ramp_time = start_and_ramp_times.second;
  return ScalarTensor::nonic_ramp_function(time, start_time, ramp_time);
}
