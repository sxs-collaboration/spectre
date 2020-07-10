// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"

#include <algorithm>
#include <array>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace Poisson::Solutions {

/// \cond
template <size_t Dim>
tuples::TaggedTuple<Tags::Field> Moustache<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, tmpl::list<Tags::Field> /*meta*/) const
    noexcept {
  auto field = make_with_value<Scalar<DataVector>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    get(field) *= x.get(d) * (1. - x.get(d));
  }
  auto norm_square = make_with_value<DataVector>(get<0>(x), 0.);
  for (size_t d = 0; d < Dim; d++) {
    norm_square += square(x.get(d) - 0.5);
  }
  get(field) *= pow(norm_square, 3. / 2.);
  return {std::move(field)};
}

template <>
tuples::TaggedTuple<
    ::Tags::deriv<Tags::Field, tmpl::size_t<1>, Frame::Inertial>>
Moustache<1>::variables(
    const tnsr::I<DataVector, 1>& x,
    tmpl::list<::Tags::deriv<Tags::Field, tmpl::size_t<1>,
                             Frame::Inertial>> /*meta*/) const noexcept {
  const auto& x_d = get<0>(x);
  tnsr::i<DataVector, 1> field_gradient{
      abs(x_d - 0.5) *
      evaluate_polynomial(std::array<double, 4>{{0.25, -3., 7.5, -5.}}, x_d)};
  return {std::move(field_gradient)};
}

template <>
tuples::TaggedTuple<
    ::Tags::deriv<Tags::Field, tmpl::size_t<2>, Frame::Inertial>>
Moustache<2>::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<::Tags::deriv<Tags::Field, tmpl::size_t<2>,
                             Frame::Inertial>> /*meta*/) const noexcept {
  auto field_gradient = make_with_value<tnsr::i<DataVector, 2>>(x, 0.);
  auto norm_square = square(get<0>(x) - 0.5) + square(get<1>(x) - 0.5);
  for (size_t d = 0; d < 2; d++) {
    const auto& x_d = x.get(d);
    const auto& x_p = x.get((d + 1) % 2);
    field_gradient.get(d) =
        sqrt(norm_square) * x_p * (1. - x_p) *
        (evaluate_polynomial(std::array<double, 4>{{0.25, -3.5, 7.5, -5.}},
                             x_d) +
         evaluate_polynomial(std::array<double, 3>{{0.25, -1., 1.}}, x_p) +
         2. * x_d * x_p - 2. * x_d * square(x_p));
  }
  return {std::move(field_gradient)};
}

template <>
tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>> Moustache<1>::variables(
    const tnsr::I<DataVector, 1>& x,
    tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const noexcept {
  const auto x1 = get<0>(x) - 0.5;
  // This polynomial is minus the laplacian of the 1D solution
  Scalar<DataVector> field_source(abs(x1) * (20. * square(x1) - 1.5));
  return {std::move(field_source)};
}

template <>
tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>> Moustache<2>::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const noexcept {
  const auto x1 = get<0>(x) - 0.5;
  const auto x2 = get<1>(x) - 0.5;
  const auto x1_square = square(x1);
  const auto x2_square = square(x2);
  const auto norm_square = x1_square + x2_square;
  // This polynomial is minus the laplacian of the 2D solution
  Scalar<DataVector> field_source(
      sqrt(norm_square) *
      (-0.5625 + 6.25 * norm_square - 6.125 * square(norm_square) +
       4.125 * square(x1_square) - 24.75 * x1_square * x2_square +
       4.125 * square(x2_square)));
  return {std::move(field_source)};
}
/// \endcond

template <size_t Dim>
void Moustache<Dim>::pup(PUP::er& /*p*/) noexcept {}

}  // namespace Poisson::Solutions

template class Poisson::Solutions::Moustache<1>;
template class Poisson::Solutions::Moustache<2>;
