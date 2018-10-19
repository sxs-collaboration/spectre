// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/FunctionOfTimeUpdater.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

template <size_t DerivOrder>
FunctionOfTimeUpdater<DerivOrder>::FunctionOfTimeUpdater(
    Averager<DerivOrder>&& averager, Controller<DerivOrder>&& controller,
    TimescaleTuner&& timescale_tuner) noexcept
    : averager_{std::move(averager)},
      controller_{std::move(controller)},
      timescale_tuner_{std::move(timescale_tuner)} {}

template <size_t DerivOrder>
void FunctionOfTimeUpdater<DerivOrder>::measure(
    double time, const DataVector& raw_q) noexcept {
  averager_.update(time, raw_q, timescale_tuner_.current_timescale());
}

template <size_t DerivOrder>
void FunctionOfTimeUpdater<DerivOrder>::modify(
    gsl::not_null<FunctionsOfTime::PiecewisePolynomial<DerivOrder>*> f_of_t,
    double time) noexcept {
  if (averager_(time)) {
    std::array<DataVector, DerivOrder + 1> q_and_derivs = averager_(time).get();
    // get the time offset due to averaging
    const double t_offset_of_qdot = time - averager_.average_time(time);
    const double t_offset_of_q =
        (averager_.using_average_0th_deriv_of_q() ? t_offset_of_qdot : 0.0);

    const DataVector control_signal =
        controller_(timescale_tuner_.current_timescale(), q_and_derivs,
                    t_offset_of_q, t_offset_of_qdot);
    f_of_t->update(averager_.last_time_updated(), control_signal);
    timescale_tuner_.update_timescale({{q_and_derivs[0], q_and_derivs[1]}});
  }
}

/// \cond
template class FunctionOfTimeUpdater<2>;
/// \endcond
