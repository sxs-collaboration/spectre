// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace domain {
namespace FunctionsOfTime {
/*!
 * \ingroup ControlSystemGroup
 * \brief Sets \f$f(t)\f$ and derivatives using cubic rational functions,
 * such that the first derivative approaches a constant and the second
 * derivative approaches zero.
 *
 * The resultant function of time is
 *
 * \f{align*}{
 *   f(t) &= f_0 + \frac{v(t-t_0)^3}{\tau^2+(t-t_0)^2},
 * \f}
 *
 * where \f$f_0\f$ is the value of the function \f$f\f$ at the initial time
 * \f$t_0\f$, and \f$v\f$ is the velocity that \f$f^\prime(t)\f$ approaches on a
 * timescale of \f$\tau\f$.
 */
class FixedSpeedCubic : public FunctionOfTime {
 public:
  FixedSpeedCubic() = default;
  FixedSpeedCubic(double initial_function_value, double initial_time,
                  double velocity, double decay_timescale);

  ~FixedSpeedCubic() override = default;
  FixedSpeedCubic(FixedSpeedCubic&&) = default;
  FixedSpeedCubic& operator=(FixedSpeedCubic&&) = default;
  FixedSpeedCubic(const FixedSpeedCubic&) = default;
  FixedSpeedCubic& operator=(const FixedSpeedCubic&) = default;

  // NOLINTNEXTLINE(google-runtime-references)
  WRAPPED_PUPable_decl_template(FixedSpeedCubic);

  explicit FixedSpeedCubic(CkMigrateMessage* /*unused*/) {}

  auto get_clone() const -> std::unique_ptr<FunctionOfTime> override;

  /// Returns the function at an arbitrary time `t`.
  std::array<DataVector, 1> func(const double t) const override {
    return func_and_derivs<0>(t);
  }
  /// Returns the function and its first derivative at an arbitrary time `t`.
  std::array<DataVector, 2> func_and_deriv(const double t) const override {
    return func_and_derivs<1>(t);
  }
  /// Returns the function and the first two derivatives at an arbitrary time
  /// `t`.
  std::array<DataVector, 3> func_and_2_derivs(const double t) const override {
    return func_and_derivs<2>(t);
  }

  /// Returns the domain of validity of the function.
  std::array<double, 2> time_bounds() const override {
    return {{initial_time_, std::numeric_limits<double>::infinity()}};
  }

  double expiration_after(const double /*time*/) const override {
    return std::numeric_limits<double>::infinity();
  }

  /// Returns the velocity that the function approaches
  double velocity() const { return velocity_; }

  /// Returns the timescale at which the function approaches a constant
  /// velocity.
  double decay_timescale() const { return sqrt(squared_decay_timescale_); }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const FixedSpeedCubic& lhs,
                         const FixedSpeedCubic& rhs);

  friend std::ostream& operator<<(  // NOLINT(readability-redundant-declaration
      std::ostream& os, const FixedSpeedCubic& fixed_speed_cubic);

  template <size_t MaxDerivReturned = 2>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const;

  double initial_function_value_{std::numeric_limits<double>::signaling_NaN()};
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double velocity_{std::numeric_limits<double>::signaling_NaN()};
  double squared_decay_timescale_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const FixedSpeedCubic& lhs, const FixedSpeedCubic& rhs);
}  // namespace FunctionsOfTime
}  // namespace domain
