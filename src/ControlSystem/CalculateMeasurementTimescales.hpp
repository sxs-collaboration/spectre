// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t DerivOrder>
class Controller;
class DataVector;
class TimescaleTuner;
/// \endcond

namespace control_system {
/*!
 * \ingroup ControlSystemGroup
 * \brief Calculate the measurement timescale based on the damping timescale,
 * update fraction, and DerivOrder of the control system
 *
 * The update timescale is \f$\tau_\mathrm{update} = \alpha_\mathrm{update}
 * \tau_\mathrm{damp}\f$ where \f$\tau_\mathrm{damp}\f$ is the damping timescale
 * (from the TimescaleTuner) and \f$\alpha_\mathrm{update}\f$ is the update
 * fraction (from the controller). For an Nth order control system, the averager
 * requires at least N measurements in order to perform its finite
 * differencing to calculate the derivatives of the control error. This implies
 * that the largest the measurement timescale can be is \f$\tau_\mathrm{m} =
 * \tau_\mathrm{update} / N\f$. To ensure that we have sufficient measurements,
 * we calculate the measurement timescales as \f$\tau_\mathrm{m} =
 * \tau_\mathrm{update} / N\f$ where \f$N\f$ is `measurements_per_update`.
 */
template <size_t DerivOrder>
DataVector calculate_measurement_timescales(
    const ::Controller<DerivOrder>& controller, const ::TimescaleTuner& tuner,
    const int measurements_per_update);
}  // namespace control_system
