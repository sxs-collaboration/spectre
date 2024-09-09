// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/ProductOfSinusoids.hpp"

#include <cmath>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace Poisson::Solutions::detail {

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::Field<DataType> /*meta*/) const {
  std::fill(field->begin(), field->end(), 1.);
  for (size_t d = 0; d < Dim; d++) {
    get(*field) *= sin(gsl::at(wave_numbers, d) * x.get(d));
  }
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    get(*field) *= std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  for (size_t d = 0; d < Dim; d++) {
    field_gradient->get(d) =
        gsl::at(wave_numbers, d) * cos(gsl::at(wave_numbers, d) * x.get(d));
    for (size_t other_d = 0; other_d < Dim; other_d++) {
      if (other_d != d) {
        field_gradient->get(d) *=
            sin(gsl::at(wave_numbers, other_d) * x.get(other_d));
      }
    }
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      field_gradient->get(d) *=
          std::complex<double>{cos(complex_phase), sin(complex_phase)};
    }
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::Flux<Tags::Field<DataType>, tmpl::size_t<Dim>,
                 Frame::Inertial> /*meta*/) const {
  const auto& field_gradient = cache->get_var(
      *this, ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                           Frame::Inertial>{});
  for (size_t d = 0; d < Dim; ++d) {
    flux_for_field->get(d) = field_gradient.get(d);
  }
}

template <typename DataType, size_t Dim>
void ProductOfSinusoidsVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> cache,
    ::Tags::FixedSource<Tags::Field<DataType>> /*meta*/) const {
  const auto& field = cache->get_var(*this, Tags::Field<DataType>{});
  get(*fixed_source_for_field) = get(field) * square(magnitude(wave_numbers));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class ProductOfSinusoidsVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
