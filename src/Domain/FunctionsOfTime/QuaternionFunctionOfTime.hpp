// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/math/quaternion.hpp>
#include <cmath>
#include <limits>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTimeHelpers.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Parallel/CharmPupable.hpp"

namespace domain::FunctionsOfTime {
/// \ingroup ComputationalDomainGroup
/// \brief A FunctionOfTime that stores quaternions for rotation map
///
/// \details This FunctionOfTime stores quaternions that will be used in the
/// time-dependent rotation map. To get the quaternion, an ODE is solved of the
/// form \f$ \dot{q} = \frac{1}{2} q \times \omega \f$ where \f$ \omega \f$ is
/// the orbital angular velocity which is retreived from a `PiecewisePolynomial`
/// in the global cache, and \f$ \times \f$ here is quaternion multiplication.
/// Different from a `PiecewisePolynomial`, only the quaternion itself is
/// stored, not any of the derivatives because the derivatives must be
/// calculated from the solved ODE at every function call. Also because
/// derivatives are not stored, what would be the `MaxDeriv` cannot be updated.
/// Before any function call, the `QuaternionFunctionOfTime` checks if it has
/// the same number of stored quaternions as the angular velocity
/// `PiecewisePolynomial` has stored omegas. This is to ensure we have all the
/// data necessary to solve the ODE at any time `t`. Other than these few, but
/// important, differences, the `QuaternionFuntionOfTime` is essentially
/// identical to the `PiecewisePolynomial`.
template <size_t MaxDerivReturned>
class QuaternionFunctionOfTime : public FunctionOfTime {
 public:
  QuaternionFunctionOfTime() = default;
  QuaternionFunctionOfTime(
      double t, std::array<DataVector, 1> initial_func,
      gsl::not_null<
          domain::FunctionsOfTime::PiecewisePolynomial<MaxDerivReturned>*>
          omega_f_of_t_ptr,
      double expiration_time) noexcept;

  ~QuaternionFunctionOfTime() override = default;
  QuaternionFunctionOfTime(QuaternionFunctionOfTime&&) noexcept = default;
  QuaternionFunctionOfTime& operator=(QuaternionFunctionOfTime&&) noexcept =
      default;
  QuaternionFunctionOfTime(const QuaternionFunctionOfTime&) = default;
  QuaternionFunctionOfTime& operator=(const QuaternionFunctionOfTime&) =
      default;

  explicit QuaternionFunctionOfTime(CkMigrateMessage* /*unused*/) {}

  auto get_clone() const noexcept -> std::unique_ptr<FunctionOfTime> override;

  // clang-tidy: google-runtime-references
  // clang-tidy: cppcoreguidelines-owning-memory,-warnings-as-errors
  WRAPPED_PUPable_decl_template(
      QuaternionFunctionOfTime<MaxDerivReturned>);  // NOLINT

  void reset_expiration_time(double next_expiration_time) noexcept;

  std::array<double, 2> time_bounds() const noexcept override {
    return {{stored_quaternions_and_times_.front().time, expiration_time_}};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  /// Returns the quaternion at an arbitrary time `t`.
  std::array<DataVector, 1> func(double t) const noexcept override;

  /// Returns the quaternion and its first derivative at an arbitrary time `t`.
  std::array<DataVector, 2> func_and_deriv(double t) const noexcept override;

  /// Returns the quaternion and the first two derivatives at an arbitrary
  /// time `t`.

  std::array<DataVector, 3> func_and_2_derivs(double t) const noexcept override;

  /// Returns the quaternion and the first three derivatives at an arbitrary
  /// time `t`.
  std::array<DataVector, 4> func_and_3_derivs(double t) const noexcept;

 private:
  mutable std::vector<StoredInfo<1>> stored_quaternions_and_times_;
  domain::FunctionsOfTime::PiecewisePolynomial<MaxDerivReturned>*
      omega_f_of_t_ptr_;
  double expiration_time_{std::numeric_limits<double>::lowest()};

  /// Solves the ODE \f$ \dot{q} = \frac{1}{2} q \times \omega \f$ at time `t`
  /// and stores the result in `quaternion_to_integrate`
  void solve_quaternion_ode(
      const double t, const double t0,
      std::array<DataVector, 1>& quaternion_to_integrate) const noexcept;

  /// Updates the `std::vector<StoredInfo>` to have the same number of stored
  /// quaternions as the `omega_f_of_t_ptr` has stored omegas. This is necessary
  /// to ensure we can solve the ODE at any time `t`
  void update_stored_info() const noexcept;

  /// Does common operations to all the `func` functions such as updating stored
  /// info, solving the ODE, and returning the normalized quaternion as a boost
  /// quaternion for easy calculations
  boost::math::quaternion<double> setup_func(double t) const noexcept;
};

/// \cond
template <size_t MaxDerivReturned>
PUP::able::PUP_ID QuaternionFunctionOfTime<MaxDerivReturned>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace domain::FunctionsOfTime
