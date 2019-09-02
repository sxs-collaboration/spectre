// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"

#include <cmath>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {
namespace Solutions {

template <size_t Dim>
ProductOfSinusoids<Dim>::ProductOfSinusoids(
    const std::array<double, Dim>& wave_numbers) noexcept
    : wave_numbers_(wave_numbers) {}

template <size_t Dim>
tuples::TaggedTuple<Tags::Field> ProductOfSinusoids<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, tmpl::list<Tags::Field> /*meta*/) const
    noexcept {
  auto field = make_with_value<Scalar<DataVector>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    field.get() *= sin(gsl::at(wave_numbers_, d) * x.get(d));
  }
  return {std::move(field)};
}

template <size_t Dim>
tuples::TaggedTuple<
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>>
ProductOfSinusoids<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::Tags::deriv<Tags::Field, tmpl::size_t<Dim>,
                             Frame::Inertial>> /*meta*/) const noexcept {
  auto field_gradient =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(x, 1.);
  for (size_t d = 0; d < Dim; d++) {
    field_gradient.get(d) *=
        gsl::at(wave_numbers_, d) * cos(gsl::at(wave_numbers_, d) * x.get(d));
    for (size_t other_d = 0; other_d < Dim; other_d++) {
      if (other_d != d) {
        field_gradient.get(d) *=
            sin(gsl::at(wave_numbers_, other_d) * x.get(other_d));
      }
    }
  }
  return {std::move(field_gradient)};
}

template <size_t Dim>
tuples::TaggedTuple<::Tags::FixedSource<Tags::Field>>
ProductOfSinusoids<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::Tags::FixedSource<Tags::Field>> /*meta*/) const noexcept {
  auto field_source = get<Tags::Field>(variables(x, tmpl::list<Tags::Field>{}));
  field_source.get() *= square(magnitude(wave_numbers_));
  return {std::move(field_source)};
}

template <size_t Dim>
void ProductOfSinusoids<Dim>::pup(PUP::er& p) noexcept {
  p | wave_numbers_;
}

}  // namespace Solutions
}  // namespace Poisson

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template class Poisson::Solutions::ProductOfSinusoids<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
