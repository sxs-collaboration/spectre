// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/SettleToConstant.hpp"

#include <cmath>

#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

FunctionsOfTime::SettleToConstant::SettleToConstant(
    const std::array<DataVector, 3>& initial_func_and_derivs,
    const double match_time, const double decay_time) noexcept
    : match_time_(match_time),
      inv_decay_time_(1.0 / decay_time) {
  // F = f(t0)
  // the constants are then computed from F and its derivs:
  // C = -dtF-dt2F*decay_time;
  // B = (C-dtF)*decay_time;
  // A = F-B;
  coef_c_ =
      -initial_func_and_derivs[1] - initial_func_and_derivs[2] * decay_time;
  coef_b_ = (coef_c_ - initial_func_and_derivs[1]) * decay_time;
  coef_a_ = initial_func_and_derivs[0] - coef_b_;
}

template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1>
FunctionsOfTime::SettleToConstant::func_and_derivs(const double t) const
    noexcept {
  static_assert(MaxDerivReturned < 3, "The maximum available derivative is 2.");

  // initialize result for the number of derivs requested
  std::array<DataVector, MaxDerivReturned + 1> result =
      make_array<MaxDerivReturned + 1>(DataVector(coef_a_.size(), 0.0));

  const double dt = t - match_time_;
  const double ex = exp(-dt * inv_decay_time_);

  gsl::at(result, 0) = coef_a_ + (coef_b_ + coef_c_ * dt) * ex;
  if (MaxDerivReturned > 0) {
    gsl::at(result, 1) = ex * (coef_c_ * (1.0 - dt * inv_decay_time_) -
                               coef_b_ * inv_decay_time_);
    if (MaxDerivReturned > 1) {
      gsl::at(result, 2) =
          ex * inv_decay_time_ *
          (coef_c_ * (inv_decay_time_ * dt - 2.0) + inv_decay_time_ * coef_b_);
    }
  }

  return result;
}

/// \cond
template std::array<DataVector, 1>
FunctionsOfTime::SettleToConstant::func_and_derivs<0>(const double) const
    noexcept;
template std::array<DataVector, 2>
FunctionsOfTime::SettleToConstant::func_and_derivs<1>(const double) const
    noexcept;
template std::array<DataVector, 3>
FunctionsOfTime::SettleToConstant::func_and_derivs<2>(const double) const
    noexcept;
/// \endcond
