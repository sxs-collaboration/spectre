// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/NumericalAlgorithms/Spectral/BallTestFunctions.hpp"

#include <cmath>
#include <numbers>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Math.hpp"

namespace BallTestFunctions {

ProductOfPolynomials::ProductOfPolynomials(const size_t pow_x,
                                           const size_t pow_y,
                                           const size_t pow_z)
    : pow_x_{pow_x}, pow_y_{pow_y}, pow_z_{pow_z} {
  ASSERT(
      pow_x_ + pow_y_ + pow_z_ + 3 <= 20,
      "Powers too large to compute factorials required for definite integral");
}

DataVector ProductOfPolynomials::operator()(const DataVector& r,
                                            const DataVector& theta,
                                            const DataVector& phi) const {
  return integer_pow(r * sin(theta) * cos(phi), static_cast<int>(pow_x_)) *
         integer_pow(r * sin(theta) * sin(phi), static_cast<int>(pow_y_)) *
         integer_pow(r * cos(theta), static_cast<int>(pow_z_));
}

double ProductOfPolynomials::operator()(const double r, const double theta,
                                        const double phi) const {
  return integer_pow(r * sin(theta) * cos(phi), static_cast<int>(pow_x_)) *
         integer_pow(r * sin(theta) * sin(phi), static_cast<int>(pow_y_)) *
         integer_pow(r * cos(theta), static_cast<int>(pow_z_));
}

DataVector ProductOfPolynomials::df_dr(const DataVector& r,
                                       const DataVector& theta,
                                       const DataVector& phi) const {
  if (pow_x_ + pow_y_ + pow_z_ == 0) {
    return DataVector{r.size(), 0.0};
  }
  return static_cast<double>(pow_x_ + pow_y_ + pow_z_) *
         pow(r, pow_x_ + pow_y_ + pow_z_ - 1) *
         integer_pow(sin(theta) * cos(phi), static_cast<int>(pow_x_)) *
         integer_pow(sin(theta) * sin(phi), static_cast<int>(pow_y_)) *
         integer_pow(cos(theta), static_cast<int>(pow_z_));
}

DataVector ProductOfPolynomials::df_dth(const DataVector& r,
                                        const DataVector& theta,
                                        const DataVector& phi) const {
  DataVector result{r.size(), 0.0};
  if (pow_x_ + pow_y_ + pow_z_ == 0) {
    return result;
  }
  const DataVector x = r * sin(theta) * cos(phi);
  const DataVector y = r * sin(theta) * sin(phi);
  const DataVector z = r * cos(theta);
  if (pow_x_ > 0) {
    result += pow_x_ * cos(phi) * integer_pow(x, static_cast<int>(pow_x_) - 1) *
              integer_pow(y, static_cast<int>(pow_y_)) *
              integer_pow(z, static_cast<int>(pow_z_) + 1);
  }
  if (pow_y_ > 0) {
    result += pow_y_ * sin(phi) * integer_pow(x, static_cast<int>(pow_x_)) *
              integer_pow(y, static_cast<int>(pow_y_) - 1) *
              integer_pow(z, static_cast<int>(pow_z_) + 1);
  }
  if (pow_z_ > 0) {
    result -= pow_z_ * r * sin(theta) *
              integer_pow(x, static_cast<int>(pow_x_)) *
              integer_pow(y, static_cast<int>(pow_y_)) *
              integer_pow(z, static_cast<int>(pow_z_) - 1);
  }
  return result;
}

DataVector ProductOfPolynomials::df_dph(const DataVector& r,
                                        const DataVector& theta,
                                        const DataVector& phi) const {
  DataVector result{r.size(), 0.0};
  if (pow_x_ + pow_y_ == 0) {
    return result;
  }
  const DataVector x = r * sin(theta) * cos(phi);
  const DataVector y = r * sin(theta) * sin(phi);
  const DataVector z = r * cos(theta);
  if (pow_x_ > 0) {
    result -= pow_x_ * integer_pow(x, static_cast<int>(pow_x_) - 1) *
              integer_pow(y, static_cast<int>(pow_y_) + 1) *
              integer_pow(z, static_cast<int>(pow_z_));
  }
  if (pow_y_ > 0) {
    result += pow_y_ * integer_pow(x, static_cast<int>(pow_x_) + 1) *
              integer_pow(y, static_cast<int>(pow_y_) - 1) *
              integer_pow(z, static_cast<int>(pow_z_));
  }
  return result / sin(theta);
}

double ProductOfPolynomials::definite_integral() const {
  if (pow_x_ % 2 == 1 or pow_y_ % 2 == 1 or pow_z_ % 2 == 1) {
    return 0.0;
  }
  if (pow_x_ == 0 and pow_y_ == 0 and pow_z_ == 0) {
    return 4.0 * std::numbers::pi / 3.0;
  }
  const size_t a = pow_x_ / 2;
  const size_t b = pow_y_ / 2;
  const size_t c = pow_z_ / 2;

  double ans = 4.0 * std::numbers::pi *
               integer_pow(2, static_cast<int>(a + b + c + 1)) *
               static_cast<double>(factorial(a + b + c + 1)) /
               static_cast<double>(factorial(pow_x_ + pow_y_ + pow_z_ + 3));
  if (pow_x_ != 0) {
    ans *= static_cast<double>(factorial(pow_x_ - 1)) /
           (integer_pow(2, static_cast<int>(a) - 1) *
            static_cast<double>(factorial(a - 1)));
  }
  if (pow_y_ != 0) {
    ans *= static_cast<double>(factorial(pow_y_ - 1)) /
           (integer_pow(2, static_cast<int>(b) - 1) *
            static_cast<double>(factorial(b - 1)));
  }
  if (pow_z_ != 0) {
    ans *= static_cast<double>(factorial(pow_z_ - 1)) /
           (integer_pow(2, static_cast<int>(c) - 1) *
            static_cast<double>(factorial(c - 1)));
  }
  return ans;
}
}  // namespace BallTestFunctions
