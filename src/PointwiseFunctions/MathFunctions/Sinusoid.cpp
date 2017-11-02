// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"

#include <cmath>

#include "Utilities/ConstantExpressions.hpp"

namespace MathFunctions {

Sinusoid::Sinusoid(const double amplitude, const double wavenumber,
                   const double phase) noexcept
    : amplitude_(amplitude), wavenumber_(wavenumber), phase_(phase) {}

double Sinusoid::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}
DataVector Sinusoid::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

double Sinusoid::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}
DataVector Sinusoid::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

double Sinusoid::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}
DataVector Sinusoid::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename T>
T Sinusoid::apply_call_operator(const T& x) const noexcept {
  return amplitude_ * sin(wavenumber_ * x + phase_);
}

template <typename T>
T Sinusoid::apply_first_deriv(const T& x) const noexcept {
  return wavenumber_ * amplitude_ * cos(wavenumber_ * x + phase_);
}

template <typename T>
T Sinusoid::apply_second_deriv(const T& x) const noexcept {
  return - amplitude_ * square(wavenumber_) * sin(wavenumber_ * x + phase_);
}

void Sinusoid::pup(PUP::er& p) {
  MathFunction<1>::pup(p);
  p | amplitude_;
  p | wavenumber_;
  p | phase_;
}
}  // namespace MathFunctions

/// \cond
PUP::able::PUP_ID MathFunctions::Sinusoid::my_PUP_ID =  // NOLINT
    0;
/// \endcond
