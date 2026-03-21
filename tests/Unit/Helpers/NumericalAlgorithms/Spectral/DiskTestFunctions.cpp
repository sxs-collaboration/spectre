// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/NumericalAlgorithms/Spectral/DiskTestFunctions.hpp"

#include <cmath>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace DiskTestFunctions {

ProductOfPolynomials::ProductOfPolynomials(const size_t pow_nx,
                                           const size_t pow_ny)
    : pow_nx_{pow_nx}, pow_ny_{pow_ny} {
  ASSERT(
      pow_nx_ / 2 + pow_ny_ / 2 + 1 <= 20,
      "Powers too large to compute factorials required for definite integral");
}

DataVector ProductOfPolynomials::operator()(const DataVector& r,
                                            const DataVector& phi) const {
  return pow(r * cos(phi), static_cast<double>(pow_nx_)) *
         pow(r * sin(phi), static_cast<double>(pow_ny_));
}

double ProductOfPolynomials::operator()(const double r,
                                        const double phi) const {
  return pow(r * cos(phi), static_cast<double>(pow_nx_)) *
         pow(r * sin(phi), static_cast<double>(pow_ny_));
}

DataVector ProductOfPolynomials::df_dr(const DataVector& r,
                                       const DataVector& phi) const {
  if (pow_nx_ + pow_ny_ == 0) {
    return DataVector{r.size(), 0.0};
  }
  return static_cast<double>(pow_nx_ + pow_ny_) *
         pow(r, pow_nx_ + pow_ny_ - 1) *
         pow(cos(phi), static_cast<double>(pow_nx_)) *
         pow(sin(phi), static_cast<double>(pow_ny_));
}

DataVector ProductOfPolynomials::df_dph(const DataVector& r,
                                        const DataVector& phi) const {
  if (pow_nx_ + pow_ny_ == 0) {
    return DataVector{r.size(), 0.0};
  }
  if (pow_nx_ == 0) {
    return static_cast<double>(pow_ny_) * pow(r, pow_ny_) * cos(phi) *
           pow(sin(phi), static_cast<double>(pow_ny_ - 1));
  }
  if (pow_ny_ == 0) {
    return -static_cast<double>(pow_nx_) * pow(r, pow_nx_) *
           pow(cos(phi), static_cast<double>(pow_nx_ - 1)) * sin(phi);
  }
  return static_cast<double>(pow_ny_) * pow(r, pow_nx_ + pow_ny_) *
             pow(cos(phi), static_cast<double>(pow_nx_ + 1)) *
             pow(sin(phi), static_cast<double>(pow_ny_ - 1)) -
         static_cast<double>(pow_nx_) * pow(r, pow_nx_ + pow_ny_) *
             pow(cos(phi), static_cast<double>(pow_nx_ - 1)) *
             pow(sin(phi), static_cast<double>(pow_ny_ + 1));
}

double ProductOfPolynomials::definite_integral() const {
  if ((pow_nx_ % 2 == 1) or (pow_ny_ % 2 == 1)) {
    return 0.0;
  }
  if (pow_nx_ + pow_ny_ == 0) {
    return std::numbers::pi;
  }
  if (pow_nx_ == 0) {
    return std::numbers::pi * pow(2.0, 1 - static_cast<int>(pow_ny_)) *
           static_cast<double>(factorial(pow_ny_ - 1)) /
           static_cast<double>(factorial(pow_ny_ / 2 - 1)) /
           static_cast<double>(factorial(pow_ny_ / 2 + 1));
  }
  if (pow_ny_ == 0) {
    return std::numbers::pi * pow(2.0, 1 - static_cast<int>(pow_nx_)) *
           static_cast<double>(factorial(pow_nx_ - 1)) /
           static_cast<double>(factorial(pow_nx_ / 2 - 1)) /
           static_cast<double>(factorial(pow_nx_ / 2 + 1));
  }
  return std::numbers::pi * pow(2.0, 2 - static_cast<int>(pow_nx_ + pow_ny_)) *
         static_cast<double>(factorial(pow_nx_ - 1)) *
         static_cast<double>(factorial(pow_ny_ - 1)) /
         static_cast<double>(factorial(pow_nx_ / 2 - 1)) /
         static_cast<double>(factorial(pow_ny_ / 2 - 1)) /
         static_cast<double>(factorial(pow_nx_ / 2 + pow_ny_ / 2 + 1));
}
}  // namespace DiskTestFunctions
