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
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Elasticity {

template <size_t Dim>
Scalar<DataVector> evaluate_potential_energy(
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) noexcept {
  auto stress = constitutive_relation.stress(strain, coordinates);
  Scalar<DataVector> pointwise_potential =
      make_with_value<Scalar<DataVector>>(coordinates, 0.);
  for (size_t i = 0; i < Dim; i++) {
    for (size_t j = 0; j < Dim; j++) {
      get(pointwise_potential) -= 0.5 * stress.get(i, j) * strain.get(i, j);
    }
  }
  return pointwise_potential;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template Scalar<DataVector> evaluate_potential_energy<DIM(data)>( \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::I<DataVector, DIM(data)>& coordinates,            \
      const ConstitutiveRelations::ConstitutiveRelation<DIM(data)>& \
          constitutive_relation) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace Elasticity
