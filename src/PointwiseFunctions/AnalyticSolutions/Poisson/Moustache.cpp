// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Moustache.hpp"

#include <algorithm>
#include <array>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {
namespace Solutions {

namespace {

tnsr::I<DataVector, 1> auxiliary_field(
    const tnsr::I<DataVector, 1>& x,
    const DataVector& precomputed_norm_square) noexcept {
  auto result = make_with_value<tnsr::I<DataVector, 1>>(x, 0.);
  const auto& x_d = get<0>(x);
  get<0>(result) = sqrt(precomputed_norm_square) *
                   evaluate_polynomial<double>({0.25, -3., 7.5, -5.}, x_d);
  return result;
}

tnsr::I<DataVector, 2> auxiliary_field(
    const tnsr::I<DataVector, 2>& x,
    const DataVector& precomputed_norm_square) noexcept {
  auto result = make_with_value<tnsr::I<DataVector, 2>>(x, 0.);
  for (size_t d = 0; d < 2; d++) {
    const auto& x_d = x.get(d);
    const auto& x_p = x.get((d + 1) % 2);
    result.get(d) = sqrt(precomputed_norm_square) * x_p * (1. - x_p) *
                    (evaluate_polynomial<double>({0.25, -3.5, 7.5, -5.}, x_d) +
                     evaluate_polynomial<double>({0.25, -1., 1.}, x_p) +
                     2. * x_d * x_p - 2. * x_d * square(x_p));
  }
  return result;
}

}  // namespace

template <size_t Dim>
tuples::TaggedTuple<Field, AuxiliaryField<Dim>> Moustache<Dim>::field_variables(
    const tnsr::I<DataVector, Dim>& x) const noexcept {
  auto field = make_with_value<Scalar<DataVector>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    get(field) *= x.get(d) * (1. - x.get(d));
  }
  auto norm_square = make_with_value<DataVector>(get<0>(x), 0.);
  for (size_t d = 0; d < Dim; d++) {
    norm_square += square(x.get(d) - 0.5);
  }
  get(field) *= pow(norm_square, 3. / 2.);
  return {std::move(field), auxiliary_field(x, norm_square)};
}

/// \cond
template <>
tuples::TaggedTuple<::Tags::Source<Field>, ::Tags::Source<AuxiliaryField<1>>>
Moustache<1>::source_variables(const tnsr::I<DataVector, 1>& x) const noexcept {
  const auto& x1 = get<0>(x);
  // This polynomial is minus the laplacian of the 1D solution
  Scalar<DataVector> field_source(
      evaluate_polynomial<double>({0.875, -8.5, 28.5, -40., 20.}, x1) /
      abs(x1 - 0.5));
  return {std::move(field_source),
          make_with_value<tnsr::I<DataVector, 1, Frame::Inertial>>(x, 0.)};
}

template <>
tuples::TaggedTuple<::Tags::Source<Field>, ::Tags::Source<AuxiliaryField<2>>>
Moustache<2>::source_variables(const tnsr::I<DataVector, 2>& x) const noexcept {
  auto norm_square = make_with_value<DataVector>(get<0>(x), 0.);
  for (size_t d = 0; d < 2; d++) {
    norm_square += square(x.get(d) - 0.5);
  }
  const auto& x1 = get<0>(x);
  const auto& x2 = get<1>(x);
  // This polynomial is minus the laplacian of the 2D solution
  Scalar<DataVector> field_source(
      evaluate_polynomial<DataVector>(
          {evaluate_polynomial<double>({0., 2., -7., 12., -11., 6., -2.}, x2),
           evaluate_polynomial<double>({2., -26.5, 65.5, -78., 39.}, x2),
           evaluate_polynomial<double>({-7., 65.5, -104.5, 78., -39.}, x2),
           evaluate_polynomial<double>({12., -78., 78.}, x2),
           evaluate_polynomial<double>({-11., 39., -39.}, x2),
           make_with_value<DataVector>(x2, 6.),
           make_with_value<DataVector>(x2, -2.)},
          x1) /
      sqrt(norm_square));
  return {std::move(field_source),
          make_with_value<tnsr::I<DataVector, 2, Frame::Inertial>>(x, 0.)};
}
/// \endcond

template <size_t Dim>
void Moustache<Dim>::pup(PUP::er& /*p*/) noexcept {}

}  // namespace Solutions
}  // namespace Poisson

template class Poisson::Solutions::Moustache<1>;
template class Poisson::Solutions::Moustache<2>;
