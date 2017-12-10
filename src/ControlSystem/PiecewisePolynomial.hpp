// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "DataStructures/DataVector.hpp"

/// \ingroup ControlSystemGroup
/// Contains functions of time to support the dual frame system.
namespace FunctionsOfTime {
/// \ingroup ControlSystemGroup
/// A function that has a piecewise-constant `MaxDeriv`th derivative.
template <size_t MaxDeriv>
class PiecewisePolynomial {
 public:
  PiecewisePolynomial(
      double t,
      std::array<DataVector, MaxDeriv + 1> initial_func_and_derivs) noexcept;

  ~PiecewisePolynomial() = default;
  PiecewisePolynomial(PiecewisePolynomial&&) noexcept = default;
  PiecewisePolynomial& operator=(PiecewisePolynomial&&) noexcept = default;
  PiecewisePolynomial(const PiecewisePolynomial&) = delete;
  PiecewisePolynomial& operator=(const PiecewisePolynomial&) = delete;

  /// Returns the function and `MaxDerivReturned` derivatives at
  /// an arbitrary time `t`.
  /// The function has multiple components.
  template <size_t MaxDerivReturned = MaxDeriv>
  std::array<DataVector, MaxDerivReturned + 1> operator()(double t) const
      noexcept;
  /// Updates the `MaxDeriv`th derivative of the function at the given time.
  /// `updated_max_deriv` is a vector of the `MaxDeriv`ths for each component
  void update(double time_of_update, DataVector updated_max_deriv) noexcept;
  /// Returns the domain of validity of the function.
  std::array<double, 2> time_bounds() const noexcept {
    return {{deriv_info_at_update_times_.front().time,
             deriv_info_at_update_times_.back().time}};
  }

 private:
  // There exists a DataVector for each deriv order that contains
  // the values of that deriv order for all components.
  using value_type = std::array<DataVector, MaxDeriv + 1>;

  // Holds information at single time at which the `MaxDeriv`th
  // derivative has been updated.
  struct DerivInfo {
    double time;
    value_type derivs_coefs;

    // Constructor is needed for use of emplace_back of a vector
    // (additionally, the constructor converts the supplied derivs to
    // coefficients for simplified polynomial evaluation.)
    DerivInfo(double t, value_type deriv) noexcept;
  };

  /// Returns a DerivInfo corresponding to the closest element in the range of
  /// DerivInfos with an update time that is less than or equal to `t`.
  /// The function throws an error if `t` is less than all DerivInfo update
  /// times. (unless `t` is just less than the earliest update time by roundoff,
  /// in which case it returns the DerivInfo at the earliest update time.)
  const DerivInfo& deriv_info_from_upper_bound(double t) const noexcept;

  std::vector<DerivInfo> deriv_info_at_update_times_;
};
}  // namespace FunctionsOfTime
