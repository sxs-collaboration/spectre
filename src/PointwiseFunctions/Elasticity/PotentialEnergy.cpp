// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity {

template <size_t Dim>
void potential_energy_density(
    const gsl::not_null<Scalar<DataVector>*> potential_energy_density,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept {
  destructive_resize_components(potential_energy_density,
                                coordinates.begin()->size());
  tnsr::II<DataVector, Dim> stress{coordinates.begin()->size()};
  constitutive_relation.stress(make_not_null(&stress), strain, coordinates);
  get(*potential_energy_density) = 0.;
  for (size_t i = 0; i < stress.size(); ++i) {
    get(*potential_energy_density) -=
        stress.multiplicity(i) * stress[i] * strain[i];
  }
  get(*potential_energy_density) *= 0.5;
}

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept {
  Scalar<DataVector> result =
      make_with_value<Scalar<DataVector>>(coordinates, 0.);
  potential_energy_density(make_not_null(&result), strain, coordinates,
                           constitutive_relation);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template void potential_energy_density<DIM(data)>(                \
      gsl::not_null<Scalar<DataVector>*> potential_energy_density,  \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::I<DataVector, DIM(data)>& coordinates,            \
      const ConstitutiveRelations::ConstitutiveRelation<DIM(data)>& \
          constitutive_relation) noexcept;                          \
  template Scalar<DataVector> potential_energy_density<DIM(data)>(  \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::I<DataVector, DIM(data)>& coordinates,            \
      const ConstitutiveRelations::ConstitutiveRelation<DIM(data)>& \
          constitutive_relation) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Elasticity
