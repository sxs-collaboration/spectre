// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"

#include <array>
#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Poisson::Solutions::detail {

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::Field<DataType> /*meta*/) const {
  get(*field) = 1. / sqrt(1. + get(dot_product(x, x))) + constant;
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    get(*field) *= std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field<DataType>, tmpl::size_t<Dim>,
                  Frame::Inertial> /*meta*/) const {
  DataType prefactor = -1. / cube(sqrt(1. + get(dot_product(x, x))));
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    prefactor *= std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
  get<0>(*field_gradient) = prefactor * get<0>(x);
  get<1>(*field_gradient) = prefactor * get<1>(x);
  get<2>(*field_gradient) = prefactor * get<2>(x);
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
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
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Field<DataType>> /*meta*/) const {
  get(*fixed_source_for_field) = 3. / pow<5>(sqrt(1. + get(dot_product(x, x))));
  if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
    get(*fixed_source_for_field) *=
        std::complex<double>{cos(complex_phase), sin(complex_phase)};
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class LorentzianVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
