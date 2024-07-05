// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/BnsInitialData/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace BnsInitialData {

void potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient) {
  ::tenex::evaluate<ti::I>(flux_for_potential,
                           (inverse_spatial_metric(ti::I, ti::J) -
                            rotational_shift_stress(ti::I, ti::J)) *
                               velocity_potential_gradient(ti::j));
}

void fluxes_on_face(
    gsl::not_null<tnsr::I<DataVector, 3>*> face_flux_for_potential,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential) {
  // potential optimization : precompute the product of the `face_normal`
  // and the `rotational_shift_stress`
  ::tenex::evaluate<ti::I>(
      face_flux_for_potential,
      velocity_potential() *
          (face_normal_vector(ti::I) -
           rotational_shift_stress(ti::I, ti::J) * face_normal(ti::j)));
}

void add_potential_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>&
        log_deriv_lapse_times_density_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  ::tenex::update(
      source_for_potential,
      (*source_for_potential)() -
          flux_for_potential(ti::I) *
              (log_deriv_lapse_times_density_over_specific_enthalpy(ti::i) +
               christoffel_contracted(ti::i)));
}

void Fluxes::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient) {
  potential_fluxes(flux_for_potential, rotational_shift_stress,
                   inverse_spatial_metric, velocity_potential_gradient);
}

void Fluxes::apply(const gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,
                   const tnsr::II<DataVector, 3>& /*inverse_spatial_metric*/,
                   const tnsr::II<DataVector, 3>& rotational_shift_stress,
                   const tnsr::i<DataVector, 3>& face_normal,
                   const tnsr::I<DataVector, 3>& face_normal_vector,
                   const Scalar<DataVector>& velocity_potential) {
  fluxes_on_face(flux_on_face, face_normal, face_normal_vector,
                 rotational_shift_stress, velocity_potential);
}

void Sources::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>&
        log_deriv_lapse_times_density_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_potential_sources(equation_for_potential,
                        log_deriv_lapse_times_density_over_specific_enthalpy,
                        christoffel_contracted, flux_for_potential);
}

}  // namespace BnsInitialData
