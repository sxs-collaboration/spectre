// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace Poisson::Solutions::detail {

template <typename DataType, size_t Dim>
void MoustacheVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/, Tags::Field /*meta*/) const {
  std::fill(field->begin(), field->end(), 1.);
  for (size_t d = 0; d < Dim; d++) {
    get(*field) *= x.get(d) * (1. - x.get(d));
  }
  auto norm_square = make_with_value<DataVector>(get<0>(x), 0.);
  for (size_t d = 0; d < Dim; d++) {
    norm_square += square(x.get(d) - 0.5);
  }
  get(*field) *= pow(norm_square, 3. / 2.);
}

template <typename DataType, size_t Dim>
void MoustacheVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const {
  if constexpr (Dim == 1) {
    const auto& x_d = get<0>(x);
    get<0>(*field_gradient) =
        abs(x_d - 0.5) *
        evaluate_polynomial(std::array<double, 4>{{0.25, -3., 7.5, -5.}}, x_d);
  } else if constexpr (Dim == 2) {
    auto norm_square = square(get<0>(x) - 0.5) + square(get<1>(x) - 0.5);
    for (size_t d = 0; d < 2; d++) {
      const auto& x_d = x.get(d);
      const auto& x_p = x.get((d + 1) % 2);
      field_gradient->get(d) =
          sqrt(norm_square) * x_p * (1. - x_p) *
          (evaluate_polynomial(std::array<double, 4>{{0.25, -3.5, 7.5, -5.}},
                               x_d) +
           evaluate_polynomial(std::array<double, 3>{{0.25, -1., 1.}}, x_p) +
           2. * x_d * x_p - 2. * x_d * square(x_p));
    }
  }
}

template <typename DataType, size_t Dim>
void MoustacheVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const {
  const auto& field_gradient = cache->get_var(
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>{});
  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <typename DataType, size_t Dim>
void MoustacheVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Field> /*meta*/) const {
  if constexpr (Dim == 1) {
    const auto x1 = get<0>(x) - 0.5;
    // This polynomial is minus the laplacian of the 1D solution
    get(*fixed_source_for_field) = abs(x1) * (20. * square(x1) - 1.5);
  } else if constexpr (Dim == 2) {
    const auto x1 = get<0>(x) - 0.5;
    const auto x2 = get<1>(x) - 0.5;
    const auto x1_square = square(x1);
    const auto x2_square = square(x2);
    const auto norm_square = x1_square + x2_square;
    // This polynomial is minus the laplacian of the 2D solution
    get(*fixed_source_for_field) =
        sqrt(norm_square) *
        (-0.5625 + 6.25 * norm_square - 6.125 * square(norm_square) +
         4.125 * square(x1_square) - 24.75 * x1_square * x2_square +
         4.125 * square(x2_square));
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class MoustacheVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector), (1, 2))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
