// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "ControlSystem/FunctionOfTime.hpp"
#include "DataStructures/DataVector.hpp"

namespace FunctionsOfTime {
/// \ingroup ControlSystemGroup
/// Given an initial function \f$f(t)\f$ and its first two derivatives
/// at the matching time \f$t_0\f$, the constant coefficients \f$A,B,C\f$
/// are computed such that the resulting function of time
/// \f$g(t)\f$ satisfies \f$g(t=t_0)=f(t=t_0)\f$ and
/// approaches a constant value for \f$t > t_0\f$ on a timescale
/// of \f$\tau\f$. The resultant
/// function is \f[ g(t) = A + (B+C(t-t_0)) e^{-(t-t_0)/\tau} \f]
/// where \f$\tau\f$=`decay_time` and \f$t_0\f$=`match_time`.

class SettleToConstant : public FunctionOfTime {
 public:
  SettleToConstant(const std::array<DataVector, 3>& initial_func_and_derivs,
                   double match_time, double decay_time) noexcept;

  ~SettleToConstant() override = default;
  SettleToConstant(SettleToConstant&&) noexcept = default;
  SettleToConstant& operator=(SettleToConstant&&) noexcept = default;
  SettleToConstant(const SettleToConstant&) = delete;
  SettleToConstant& operator=(const SettleToConstant&) = delete;

  /// Returns the function at an arbitrary time `t`.
  std::array<DataVector, 1> func(double t) const noexcept override {
    return func_and_derivs<0>(t);
  }
  /// Returns the function and its first derivative at an arbitrary time `t`.
  std::array<DataVector, 2> func_and_deriv(double t) const noexcept override {
    return func_and_derivs<1>(t);
  }
  /// Returns the function and the first two derivatives at an arbitrary time
  /// `t`.
  std::array<DataVector, 3> func_and_2_derivs(double t) const
      noexcept override {
    return func_and_derivs<2>(t);
  }

  /// Returns the domain of validity of the function.
  std::array<double, 2> time_bounds() const noexcept override {
    return {{match_time_, std::numeric_limits<double>::max()}};
  }

 private:
  template <size_t MaxDerivReturned = 2>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const
      noexcept;

  DataVector coef_a_, coef_b_, coef_c_;
  double match_time_;
  double inv_decay_time_;
};
}  // namespace FunctionsOfTime
