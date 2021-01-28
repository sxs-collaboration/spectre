// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace Poisson::Solutions::detail {

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> field,
    const gsl::not_null<Cache*> /*cache*/, Tags::Field /*meta*/) const
    noexcept {
  get(*field) = 1. / sqrt(1. + get(dot_product(x, x)));
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*> field_gradient,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::Field, tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
    const noexcept {
  const DataVector prefactor = -1. / cube(sqrt(1. + get(dot_product(x, x))));
  get<0>(*field_gradient) = prefactor * get<0>(x);
  get<1>(*field_gradient) = prefactor * get<1>(x);
  get<2>(*field_gradient) = prefactor * get<2>(x);
}

template <typename DataType, size_t Dim>
void LorentzianVariables<DataType, Dim>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_field,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Field> /*meta*/) const noexcept {
  get(*fixed_source_for_field) = 3. / pow<5>(sqrt(1. + get(dot_product(x, x))));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class LorentzianVariables<DTYPE(data), DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector), (3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace Poisson::Solutions::detail
/// \endcond
