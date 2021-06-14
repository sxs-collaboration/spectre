// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"

#include <cmath>
#include <memory>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::FunctionsOfTime {
FixedSpeedCubic::FixedSpeedCubic(const double initial_function_value,
                                 const double initial_time,
                                 const double velocity,
                                 const double decay_timescale) noexcept
    : initial_function_value_(initial_function_value),
      initial_time_(initial_time),
      velocity_(velocity),
      squared_decay_timescale_(square(decay_timescale)) {}

std::unique_ptr<FunctionOfTime> FixedSpeedCubic::get_clone() const noexcept {
  return std::make_unique<FixedSpeedCubic>(*this);
}

template <size_t MaxDerivReturned>
std::array<DataVector, MaxDerivReturned + 1> FixedSpeedCubic::func_and_derivs(
    const double t) const noexcept {
  static_assert(MaxDerivReturned < 3, "The maximum available derivative is 2.");

  // initialize result for the number of derivs requested
  std::array<DataVector, MaxDerivReturned + 1> result =
      make_array<MaxDerivReturned + 1>(DataVector(1, 0.0));

  const double dt = t - initial_time_;
  const double denom = squared_decay_timescale_ + square(dt);
  ASSERT(fabs(denom) > 0.0,
         "FixedSpeedCubic denominator should not be zero; if evaluating at "
         "t == t0, then do not set the decay timescale tau to 0.");
  const double one_over_denom = 1.0 / (squared_decay_timescale_ + square(dt));

  gsl::at(result, 0)[0] =
      initial_function_value_ + velocity_ * cube(dt) * one_over_denom;
  if (MaxDerivReturned > 0) {
    gsl::at(result, 1)[0] = (3.0 * squared_decay_timescale_ + square(dt)) *
                            square(dt) * velocity_ * square(one_over_denom);
    if (MaxDerivReturned > 1) {
      gsl::at(result, 2)[0] = -2.0 * squared_decay_timescale_ *
                              (3.0 * squared_decay_timescale_ - square(dt)) *
                              dt * velocity_ * cube(one_over_denom);
    }
  }

  return result;
}

void FixedSpeedCubic::pup(PUP::er& p) {
  FunctionOfTime::pup(p);
  p | initial_function_value_;
  p | initial_time_;
  p | velocity_;
  p | squared_decay_timescale_;
}

bool operator==(const FixedSpeedCubic& lhs,
                const FixedSpeedCubic& rhs) noexcept {
  return lhs.initial_function_value_ == rhs.initial_function_value_ and
         lhs.initial_time_ == rhs.initial_time_ and
         lhs.velocity_ == rhs.velocity_ and
         lhs.squared_decay_timescale_ == rhs.squared_decay_timescale_;
}

bool operator!=(const FixedSpeedCubic& lhs,
                const FixedSpeedCubic& rhs) noexcept {
  return not(lhs == rhs);
}

PUP::able::PUP_ID FixedSpeedCubic::my_PUP_ID = 0;  // NOLINT

#define DERIV(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                       \
  template std::array<DataVector, DERIV(data) + 1> \
  FixedSpeedCubic::func_and_derivs<DERIV(data)>(const double) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2))

#undef DERIV
#undef INSTANTIATE
}  // namespace domain::FunctionsOfTime
