// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Krivodonova.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Math.hpp"

namespace ScalarAdvection::AnalyticData {

template <typename T>
Scalar<T> Krivodonova::u(const tnsr::I<T, 1>& x) const noexcept {
  const double a{0.5};
  const double z{-0.7};
  const double delta{0.005};
  const double alpha{10.0};
  const double beta{log(2.0) / (36.0 * pow(delta, 2.0))};

  const auto F = [](const double x_var, const double alpha_var,
                    const double a_var) noexcept {
    return sqrt(fmax(1.0 - pow(alpha_var, 2.0) * pow(x_var - a_var, 2.0), 0.0));
  };
  const auto G = [](const double x_var, const double beta_var,
                    const double z_var) noexcept {
    return exp(-beta_var * pow(x_var - z_var, 2.0));
  };

  Scalar<T> u_variable{get<0>(x)};
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    const auto& x0 = get<0>(x)[i];
    auto& ui = get(u_variable)[i];

    if ((-0.8 <= x0) and (x0 <= -0.6)) {
      ui = (G(x0, beta, z - delta) + G(x0, beta, z + delta) +
            4.0 * G(x0, beta, z)) /
           6.0;
    } else if ((-0.4 <= x0) and (x0 <= -0.2)) {
      ui = 1.0;
    } else if ((0.0 <= x0) and (x0 <= 0.2)) {
      ui = 1.0 - abs(10.0 * x0 - 1.0);
    } else if ((0.4 <= x0) and (x0 <= 0.6)) {
      ui = (F(x0, alpha, a - delta) + F(x0, alpha, a + delta) +
            4.0 * F(x0, alpha, a)) /
           6.0;
    } else {
      ui = 0.0;
    }
  }
  return u_variable;
}

tuples::TaggedTuple<Tags::U> Krivodonova::variables(
    const tnsr::I<DataVector, 1>& x,
    tmpl::list<Tags::U> /*meta*/) const noexcept {
  return {u(x)};
}

void Krivodonova::pup(PUP::er& /*p*/) noexcept {}

bool operator==(const Krivodonova& /*lhs*/,
                const Krivodonova& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const Krivodonova& lhs, const Krivodonova& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace ScalarAdvection::AnalyticData
