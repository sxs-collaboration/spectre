// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/BentBeam.hpp"

#include <algorithm>
#include <array>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity::Solutions::detail {

template <typename DataType>
void BentBeamVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 2>*> displacement,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::Displacement<2> /*meta*/) const {
  const double youngs_modulus = constitutive_relation.youngs_modulus();
  const double poisson_ratio = constitutive_relation.poisson_ratio();
  const double prefactor =
      12. * bending_moment / (youngs_modulus * cube(height));
  get<0>(*displacement) = -prefactor * get<0>(x) * get<1>(x);
  get<1>(*displacement) =
      prefactor / 2. *
      (square(get<0>(x)) + poisson_ratio * square(get<1>(x)) -
       square(length) / 4.);
}

template <typename DataType>
void BentBeamVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 2>*> strain,
    const gsl::not_null<Cache*> /*cache*/, Tags::Strain<2> /*meta*/) const {
  const double youngs_modulus = constitutive_relation.youngs_modulus();
  const double poisson_ratio = constitutive_relation.poisson_ratio();
  const double prefactor =
      12. * bending_moment / (youngs_modulus * cube(height));
  get<0, 0>(*strain) = -prefactor * get<1>(x);
  get<1, 1>(*strain) = prefactor * poisson_ratio * get<1>(x);
  get<0, 1>(*strain) = 0.;
}

template <typename DataType>
void BentBeamVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 2>*> minus_stress,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::MinusStress<2> /*meta*/) const {
  get<0, 0>(*minus_stress) = -12. * bending_moment / cube(height) * get<1>(x);
  get<1, 1>(*minus_stress) = 0.;
  get<0, 1>(*minus_stress) = 0.;
}

template <typename DataType>
void BentBeamVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> potential_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::PotentialEnergyDensity<2> /*meta*/) const {
  get(*potential_energy_density) = 72. * square(bending_moment) /
                                   constitutive_relation.youngs_modulus() /
                                   pow<6>(height) * square(get<1>(x));
}

template <typename DataType>
void BentBeamVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 2>*> fixed_source_for_displacement,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Displacement<2>> /*meta*/) const {
  std::fill(fixed_source_for_displacement->begin(),
            fixed_source_for_displacement->end(), 0.);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class BentBeamVariables<DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Elasticity::Solutions::detail
