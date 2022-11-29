// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/CalculateMeasurementTimescales.hpp"

#include "ControlSystem/Controller.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"

namespace control_system {
template <size_t DerivOrder>
DataVector calculate_measurement_timescales(
    const ::Controller<DerivOrder>& controller, const ::TimescaleTuner& tuner,
    const int measurements_per_update) {
  return tuner.current_timescale() * controller.get_update_fraction() /
         static_cast<double>(measurements_per_update);
}

template DataVector calculate_measurement_timescales<2>(
    const ::Controller<2>& controller, const ::TimescaleTuner& tuner,
    const int measurements_per_update);
template DataVector calculate_measurement_timescales<3>(
    const ::Controller<3>& controller, const ::TimescaleTuner& tuner,
    const int measurements_per_update);

}  // namespace control_system
