// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace MathFunctions {

Gaussian::Gaussian(const double amplitude, const double width,
                   const double center) noexcept
    : amplitude_(amplitude), inverse_width_(1.0 / width), center_(center) {}

double Gaussian::operator()(const double& x) const noexcept {
  return apply_call_operator(x);
}
DataVector Gaussian::operator()(const DataVector& x) const noexcept {
  return apply_call_operator(x);
}

double Gaussian::first_deriv(const double& x) const noexcept {
  return apply_first_deriv(x);
}
DataVector Gaussian::first_deriv(const DataVector& x) const noexcept {
  return apply_first_deriv(x);
}

double Gaussian::second_deriv(const double& x) const noexcept {
  return apply_second_deriv(x);
}
DataVector Gaussian::second_deriv(const DataVector& x) const noexcept {
  return apply_second_deriv(x);
}

double Gaussian::third_deriv(const double& x) const noexcept {
  return apply_third_deriv(x);
}
DataVector Gaussian::third_deriv(const DataVector& x) const noexcept {
  return apply_third_deriv(x);
}

template <typename T>
T Gaussian::apply_call_operator(const T& x) const noexcept {
  return amplitude_ * exp(-square((x - center_) * inverse_width_));
}

template <typename T>
T Gaussian::apply_first_deriv(const T& x) const noexcept {
  return (-2.0 * amplitude_ * square(inverse_width_)) * (x - center_) *
         exp(-square((x - center_) * inverse_width_));
}

template <typename T>
T Gaussian::apply_second_deriv(const T& x) const noexcept {
  return (-2.0 * amplitude_ * square(inverse_width_)) *
         (1.0 - 2.0 * square(x - center_) * square(inverse_width_)) *
         exp(-square((x - center_) * inverse_width_));
}

template <typename T>
T Gaussian::apply_third_deriv(const T& x) const noexcept {
  return 4.0 * amplitude_ * pow<4>(inverse_width_) * (x - center_) *
         (3.0 - 2.0 * square((x - center_) * inverse_width_)) *
         exp(-square((x - center_) * inverse_width_));
}

void Gaussian::pup(PUP::er& p) {
  MathFunction<1>::pup(p);
  p | amplitude_;
  p | inverse_width_;
  p | center_;
}

bool operator==(const Gaussian& lhs, const Gaussian& rhs) noexcept {
  return lhs.amplitude_ == rhs.amplitude_ and
         lhs.inverse_width_ == rhs.inverse_width_ and
         lhs.center_ == rhs.center_;
}

bool operator!=(const Gaussian& lhs, const Gaussian& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace MathFunctions

/// \cond
PUP::able::PUP_ID MathFunctions::Gaussian::my_PUP_ID =  // NOLINT
    0;
/// \endcond
