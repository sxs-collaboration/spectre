// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/PowX.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/MakeWithValue.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace MathFunctions {

PowX::PowX(const int power) noexcept : power_(power) {}

double PowX::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}
DataVector PowX::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

double PowX::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}
DataVector PowX::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

double PowX::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}
DataVector PowX::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

template <typename T>
T PowX::apply_call_operator(const T& x) const noexcept {
  return pow(x, power_);
}

template <typename T>
T PowX::apply_first_deriv(const T& x) const noexcept {
  return 0 == power_ ? make_with_value<T>(x, 0.0)
                     : power_ * pow(x, power_ - 1);
}

template <typename T>
T PowX::apply_second_deriv(const T& x) const noexcept {
  return 0 == power_ or 1 == power_
             ? make_with_value<T>(x, 0.0)
             : power_ * (power_ - 1) * pow(x, power_ - 2);
}

void PowX::pup(PUP::er& p) {
  MathFunction<1>::pup(p);
  p | power_;
}
}  // namespace MathFunctions

/// \cond
PUP::able::PUP_ID MathFunctions::PowX::my_PUP_ID =  // NOLINT
    0;
/// \endcond
