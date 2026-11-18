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
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace Elasticity {

template <size_t Dim>
void potential_energy_density(const gsl::not_null<Scalar<DataVector>*> result,
                              const tnsr::ii<DataVector, Dim>& strain,
                              const tnsr::II<DataVector, Dim>& stress) {
  set_number_of_grid_points(result, strain);
  get(*result) = 0.;
  for (size_t i = 0; i < stress.size(); ++i) {
    get(*result) -= stress.multiplicity(i) * stress[i] * strain[i];
  }
  get(*result) *= 0.5;
}

template <size_t Dim>
void potential_energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) {
  tnsr::II<DataVector, Dim> stress{coordinates.begin()->size()};
  constitutive_relation.stress(make_not_null(&stress), strain, coordinates);
  potential_energy_density(result, strain, stress);
}

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::I<DataVector, Dim>& coordinates,
    const ConstitutiveRelations::ConstitutiveRelation<Dim>&
        constitutive_relation) {
  Scalar<DataVector> result =
      make_with_value<Scalar<DataVector>>(coordinates, 0.);
  potential_energy_density(make_not_null(&result), strain, coordinates,
                           constitutive_relation);
  return result;
}

template <size_t Dim>
void bulk_potential_energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::II<DataVector, Dim>& stress) {
  set_number_of_grid_points(result, strain);

  // Traces Theta = S_ii and Sigma = T^ii
  DataVector trace_strain = strain.get(0, 0);
  DataVector trace_stress = stress.get(0, 0);
  for (size_t d = 1; d < Dim; ++d) {
    trace_strain += strain.get(d, d);
    trace_stress += stress.get(d, d);
  }

  // U_bulk = -Theta Sigma / (2 Dim)
  get(*result) = -1.0 / (2.0 * Dim) * trace_strain * trace_stress;
}

template <size_t Dim>
void shear_potential_energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, Dim>& strain,
    const tnsr::II<DataVector, Dim>& stress) {
  set_number_of_grid_points(result, strain);

  // Traces Theta = S_ii and Sigma = T^ii
  DataVector trace_strain = strain.get(0, 0);
  DataVector trace_stress = stress.get(0, 0);
  for (size_t d = 1; d < Dim; ++d) {
    trace_strain += strain.get(d, d);
    trace_stress += stress.get(d, d);
  }

  // U_shear = -1/2 e_ij s^ij, contracting the deviators
  // e_ij = S_ij - Theta / Dim delta_ij and s^ij = T^ij - Sigma / Dim delta^ij
  // over the upper triangle (factor 2 off the diagonal). This does not form
  // S_ij T^ij, so it avoids the cancellation of total minus bulk.
  get(*result) = 0.;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      DataVector dev_strain = strain.get(i, j);
      DataVector dev_stress = stress.get(i, j);
      if (i == j) {
        dev_strain -= trace_strain / Dim;
        dev_stress -= trace_stress / Dim;
      }
      const double mult = (i == j) ? 1.0 : 2.0;
      get(*result) -= mult * dev_stress * dev_strain;
    }
  }
  get(*result) *= 0.5;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template void potential_energy_density<DIM(data)>(                \
      gsl::not_null<Scalar<DataVector>*> result,                    \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::II<DataVector, DIM(data)>& stress);               \
  template void potential_energy_density<DIM(data)>(                \
      gsl::not_null<Scalar<DataVector>*> result,                    \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::I<DataVector, DIM(data)>& coordinates,            \
      const ConstitutiveRelations::ConstitutiveRelation<DIM(data)>& \
          constitutive_relation);                                   \
  template Scalar<DataVector> potential_energy_density<DIM(data)>(  \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::I<DataVector, DIM(data)>& coordinates,            \
      const ConstitutiveRelations::ConstitutiveRelation<DIM(data)>& \
          constitutive_relation);                                   \
  template void bulk_potential_energy_density<DIM(data)>(           \
      gsl::not_null<Scalar<DataVector>*> result,                    \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::II<DataVector, DIM(data)>& stress);               \
  template void shear_potential_energy_density<DIM(data)>(          \
      gsl::not_null<Scalar<DataVector>*> result,                    \
      const tnsr::ii<DataVector, DIM(data)>& strain,                \
      const tnsr::II<DataVector, DIM(data)>& stress);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Elasticity
