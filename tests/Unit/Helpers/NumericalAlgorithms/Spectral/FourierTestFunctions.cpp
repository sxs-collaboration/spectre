// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/NumericalAlgorithms/Spectral/FourierTestFunctions.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace FourierTestFunctions {

ProductOfPolynomials::ProductOfPolynomials(const size_t pow_nx,
                                           const size_t pow_ny)
    : pow_nx_{pow_nx}, pow_ny_{pow_ny} {}

DataVector ProductOfPolynomials::operator()(const DataVector& phi) const {
  return pow(cos(phi), static_cast<double>(pow_nx_)) *
         pow(sin(phi), static_cast<double>(pow_ny_));
}

double ProductOfPolynomials::operator()(const double phi) const {
  return pow(cos(phi), static_cast<double>(pow_nx_)) *
         pow(sin(phi), static_cast<double>(pow_ny_));
}

DataVector ProductOfPolynomials::df_dph(const DataVector& phi) const {
  if (pow_nx_ + pow_ny_ == 0) {
    return DataVector{phi.size(), 0.0};
  }
  if (pow_nx_ == 0) {
    return static_cast<double>(pow_ny_) * cos(phi) *
           pow(sin(phi), static_cast<double>(pow_ny_ - 1));
  }
  if (pow_ny_ == 0) {
    return -static_cast<double>(pow_nx_) *
           pow(cos(phi), static_cast<double>(pow_nx_ - 1)) * sin(phi);
  }
  return static_cast<double>(pow_ny_) *
             pow(cos(phi), static_cast<double>(pow_nx_ + 1)) *
             pow(sin(phi), static_cast<double>(pow_ny_ - 1)) -
         static_cast<double>(pow_nx_) *
             pow(cos(phi), static_cast<double>(pow_nx_ - 1)) *
             pow(sin(phi), static_cast<double>(pow_ny_ + 1));
}

double ProductOfPolynomials::definite_integral() const {
  if ((pow_nx_ % 2 == 1) or (pow_ny_ % 2 == 1)) {
    return 0.0;
  }
  double product = 1.0;
  double m = 0.0;
  for (size_t i = 1; i <= pow_nx_ / 2; ++i) {
    m += 2.0;
    product *= (m - 1.0) / m;
  }
  double n = 0.0;
  for (size_t j = 1; j <= pow_ny_ / 2; ++j) {
    n += 2.0;
    product *= (n - 1.0) / (m + n);
  }

  return 2.0 * M_PI * product;
}

DataVector ProductOfPolynomials::modes() const {
  const size_t p = pow_nx_;
  const size_t q = pow_ny_;
  const size_t j = p + q;
  DataVector result{2 * j + 1, 0.0};
  const bool q_is_odd = q % 2 == 1;
  const double overall_sign = (q / 2) % 2 == 0 ? 1.0 : -1.0;
  const auto two_to_the_j = static_cast<double>(two_to_the(j));
  for (size_t s = 0; s <= p; ++s) {
    const auto bin_p_s = static_cast<double>(binomial(p, s));
    for (size_t l = 0; l <= q; ++l) {
      const auto bin_q_l = static_cast<double>(binomial(q, l));
      const double sign = (q - l) % 2 == 0 ? 1.0 : -1.0;
      const int m = static_cast<int>(2 * s + 2 * l) - static_cast<int>(j);
      const size_t storage_index =
          Spectral::Fourier::modal_storage_index(q_is_odd ? -abs(m) : abs(m));
      const double parity = (q_is_odd and m < 0) ? -1.0 : 1.0;
      result[storage_index] += parity * sign * bin_q_l * bin_p_s;
    }
  }
  result *= (overall_sign / two_to_the_j);
  return result;
}
}  // namespace FourierTestFunctions
