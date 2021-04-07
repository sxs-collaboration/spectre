// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/SettleToConstant.hpp"

#include <cmath>
#include <memory>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::FunctionsOfTime {
SettleToConstant::SettleToConstant(
    const std::array<DataVector, 3>& initial_func_and_derivs,
    const double match_time, const double decay_time) noexcept
    : match_time_(match_time), inv_decay_time_(1.0 / decay_time) {
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

std::unique_ptr<FunctionOfTime> SettleToConstant::get_clone() const noexcept {
  return std::make_unique<SettleToConstant>(*this);
}

template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1> SettleToConstant::func_and_derivs(
    const double t) const noexcept {
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

void SettleToConstant::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  p | coef_a_;
  p | coef_b_;
  p | coef_c_;
  p | match_time_;
  p | inv_decay_time_;
}

bool operator==(const SettleToConstant& lhs,
                const SettleToConstant& rhs) noexcept {
  return lhs.coef_a_ == rhs.coef_a_ and lhs.coef_b_ == rhs.coef_b_ and
         lhs.coef_c_ == rhs.coef_c_ and lhs.match_time_ == rhs.match_time_ and
         lhs.inv_decay_time_ == rhs.inv_decay_time_;
}

bool operator!=(const SettleToConstant& lhs,
                const SettleToConstant& rhs) noexcept {
  return not(lhs == rhs);
}

PUP::able::PUP_ID SettleToConstant::my_PUP_ID = 0;  // NOLINT

#define DERIV(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                       \
  template std::array<DataVector, DERIV(data) + 1> \
  SettleToConstant::func_and_derivs<DERIV(data)>(const double) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef DERIV
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
