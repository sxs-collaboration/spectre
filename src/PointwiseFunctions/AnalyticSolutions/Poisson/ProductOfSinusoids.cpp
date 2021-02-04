// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

/// \cond
namespace Poisson::Solutions::detail {

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/, Tags::Field /*meta*/) const
    noexcept {
  std::fill(field->begin(), field->end(), 1.);
  for (size_t d = 0; d < Dim; d++) {
    get(*field) *= sin(gsl::at(wave_numbers, d) * x.get(d));
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const noexcept {
  for (size_t d = 0; d < Dim; d++) {
    field_gradient->get(d) =
        gsl::at(wave_numbers, d) * cos(gsl::at(wave_numbers, d) * x.get(d));
    for (size_t other_d = 0; other_d < Dim; other_d++) {
      if (other_d != d) {
        field_gradient->get(d) *=
            sin(gsl::at(wave_numbers, other_d) * x.get(other_d));
      }
    }
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& field_gradient = cache->get_var(
      ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial>{});
  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::FixedSource<Tags::Field> /*meta*/) const noexcept {
  const auto& field = cache->get_var(Tags::Field{});
  get(*fixed_source_for_field) = get(field) * square(magnitude(wave_numbers));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class ProductOfSinusoidsVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector), (1, 2, 3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
/// \endcond
