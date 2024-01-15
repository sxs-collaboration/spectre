// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace domain::FunctionsOfTime {
/// \ingroup ControlSystemGroup
/// \brief Given an initial function of time that is a unit quaternion,
/// transitions to a constant-in-time unit quaternion.
///
/// Given an initial unit quaternion \f$\mathbf{f}(t)\f$ and its first two
/// derivatives at the matching time \f$t_0\f$, return a function of time
/// \f$\mathbf{g}(t)\f$ satisfies \f$\mathbf{g}(t=t_0)=\mathbf{f}f(t=t_0)\f$ and
/// approaches a constant value for \f$t > t_0\f$ on a timescale of \f$\tau\f$.
/// This is done by internally holding a `SettleToConstant` function of time
/// initialized from \f$\mathbf{f}(t)\f$ and its first two derivatives at the
/// matching time, but then ensuring that \f$\mathbf{g}(t)\f$ remains
/// a unit quaternion.
class SettleToConstantQuaternion : public FunctionOfTime {
 public:
  SettleToConstantQuaternion() = default;
  SettleToConstantQuaternion(
      const std::array<DataVector, 3>& initial_func_and_derivs,
      double match_time, double decay_time);

  ~SettleToConstantQuaternion() override = default;
  SettleToConstantQuaternion(SettleToConstantQuaternion&&) = default;
  SettleToConstantQuaternion& operator=(SettleToConstantQuaternion&&) = default;
  SettleToConstantQuaternion(const SettleToConstantQuaternion&) = default;
  SettleToConstantQuaternion& operator=(const SettleToConstantQuaternion&) =
      default;

  // NOLINTNEXTLINE(google-runtime-references)
  WRAPPED_PUPable_decl_template(SettleToConstantQuaternion);

  explicit SettleToConstantQuaternion(CkMigrateMessage* /*unused*/) {}

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
    return {{match_time_, std::numeric_limits<double>::infinity()}};
  }

  double expiration_after(const double /*time*/) const override {
    return std::numeric_limits<double>::infinity();
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const SettleToConstantQuaternion& lhs,
                         const SettleToConstantQuaternion& rhs);

  template <size_t MaxDerivReturned = 2>
  std::array<DataVector, MaxDerivReturned + 1> func_and_derivs(double t) const;

  SettleToConstant unnormalized_function_of_time_;
  double match_time_{std::numeric_limits<double>::signaling_NaN()};
  double decay_time_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator!=(const SettleToConstantQuaternion& lhs,
                const SettleToConstantQuaternion& rhs);
}  // namespace domain::FunctionsOfTime
