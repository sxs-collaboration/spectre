// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Krivodonova.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarAdvection::Solutions {

template <typename DataType>
tuples::TaggedTuple<ScalarAdvection::Tags::U> Krivodonova::variables(
    const tnsr::I<DataType, 1>& x, double t,
    tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const noexcept {
  // map each grid points x(t) back to its initial position within [-1, 1] at
  // t=0
  auto x0 = make_with_value<tnsr::I<DataType, 1>>(x, 0.0);
  get<0>(x0) = get<0>(x) - t;
  for (size_t i = 0; i < get_size(get<0>(x)); ++i) {
    auto& xi = get<0>(x0)[i];
    // Since we are using the periodic boundary condition with the domain
    // [-1.0, 1.0], we need to do a 'modulo' operation:
    //
    //  x - vt = 2.0 * N + x0
    //
    // and get the value of x0, where N is an integer and x0 is a real number in
    // [-1.0, 1.0). Then x0 corresponds to the initial position of the point
    // (x,t). We use the value of x0 to compute U(x,t) = U(x0,0).
    xi = xi - 2.0 * floor(0.5 * (xi + 1.0));
  }

  // parameters for the initial profile (from the Krivodonova paper)
  const double a{0.5};
  const double z{-0.7};
  const double delta{0.005};
  const double alpha{10.0};
  const double beta{log(2.0) / (36.0 * square(delta))};

  const auto F = [](const double x_var, const double alpha_var,
                    const double a_var) noexcept {
    return sqrt(fmax(1.0 - pow(alpha_var, 2.0) * pow(x_var - a_var, 2.0), 0.0));
  };
  const auto G = [](const double x_var, const double beta_var,
                    const double z_var) noexcept {
    return exp(-beta_var * pow(x_var - z_var, 2.0));
  };

  // evaluate U(x,t) = U(x0,0)
  auto u_variable = make_with_value<Scalar<DataType>>(x0, 0.0);
  for (size_t i = 0; i < get_size(get<0>(x0)); ++i) {
    const auto& xi = get<0>(x0)[i];
    auto& ui = get(u_variable)[i];

    if ((-0.8 <= xi) and (xi <= -0.6)) {
      ui = (G(xi, beta, z - delta) + G(xi, beta, z + delta) +
            4.0 * G(xi, beta, z)) /
           6.0;
    } else if ((-0.4 <= xi) and (xi <= -0.2)) {
      ui = 1.0;
    } else if ((0.0 <= xi) and (xi <= 0.2)) {
      ui = 1.0 - fabs(10.0 * xi - 1.0);
    } else if ((0.4 <= xi) and (xi <= 0.6)) {
      ui = (F(xi, alpha, a - delta) + F(xi, alpha, a + delta) +
            4.0 * F(xi, alpha, a)) /
           6.0;
    } else {
      ui = 0.0;
    }
  }
  return u_variable;
}

void Krivodonova::pup(PUP::er& /*p*/) noexcept {}

bool operator==(const Krivodonova& /*lhs*/,
                const Krivodonova& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const Krivodonova& lhs, const Krivodonova& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace ScalarAdvection::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template tuples::TaggedTuple<ScalarAdvection::Tags::U> \
  ScalarAdvection::Solutions::Krivodonova::variables(    \
      const tnsr::I<DTYPE(data), 1>& x, double t,        \
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))

#undef DTYPE
#undef INSTANTIATE
