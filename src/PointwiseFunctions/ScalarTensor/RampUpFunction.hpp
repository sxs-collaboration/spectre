// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

namespace ScalarTensor {
/// @{
/*!
 * \brief Function to smoothly turn on the coupling terms.
 *
 * \details Implements Eq. B4 of \cite Okounkova:2019zjf :
 * \begin{align}
 *    t_*  &\equiv (t - t_\mathrm{s})/t_\mathrm{ramp} \,, \\
 *    F(t) &= t_*^5  (126 + t_* (-420 + t_* (540 + t_* (-315 + 70 t_*))))~,
 * \end{align}
 * where $ t_\mathrm{s} $ and $ t_\mathrm{ramp} $ are parameters that
 control the start time and duration of the turn-on period.
 *
 */
double nonic_ramp_function(double time, double start_time, double ramp_time);

double nonic_ramp_function(double time,
                             std::pair<double, double> start_and_ramp_times);
/// @}
}  // namespace ScalarTensor
