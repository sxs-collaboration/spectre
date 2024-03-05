// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/IrrotationalBns/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/IrrotationalBns/Geometry.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace IrrotationalBns {

void flat_potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  ::tenex::evaluate<ti::I>(
      flux_for_potential,
      -auxiliary_velocity(ti::j) * rotational_shift_stress(ti::I, ti::J));
  for (size_t d = 0; d < 3; d++) {
    flux_for_potential->get(d) += auxiliary_velocity.get(d);
  }
}

void curved_potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  ::tenex::evaluate<ti::I>(flux_for_potential,
                           (inverse_spatial_metric(ti::I, ti::J) -
                            rotational_shift_stress(ti::I, ti::J)) *
                               auxiliary_velocity(ti::j));
}

void fluxes_on_face(
    gsl::not_null<tnsr::I<DataVector, 3>*> face_flux_for_potential,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential) {
  ::tenex::evaluate<ti::I>(
      face_flux_for_potential,
      velocity_potential() *
          (face_normal_vector(ti::I) -
           face_normal(ti::j) * rotational_shift_stress(ti::I, ti::J)));
}

void add_flat_potential_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_lapse_over_specific_enthalpy,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  for (size_t i = 0; i < 3; i++) {
    source_for_potential->get() -=
        flux_for_potential.get(i) *
        log_deriv_lapse_over_specific_enthalpy.get(i);
  }
}

void add_curved_potential_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  // The flux for the potential is the auxiliary velocity
  // (when the index is in the right place)
  ::tenex::update(source_for_potential,
                  (*source_for_potential)() +
                      flux_for_potential(ti::I) *
                          (log_deriv_of_lapse_over_specific_enthalpy(ti::i) -
                           christoffel_contracted(ti::i)));
}
// Flat-space cartesian fluxes

void Fluxes<Geometry::FlatCartesian>::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  flat_potential_fluxes(flux_for_potential, rotational_shift_stress,
                        auxiliary_velocity);
}

void Fluxes<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,

    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const Scalar<DataVector>& velocity_potential) {
  fluxes_on_face(flux_on_face, face_normal_vector, face_normal,
                 rotational_shift_stress, velocity_potential);
}

// Curved-space (generic fluxes)

void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  curved_potential_fluxes(flux_for_potential, rotational_shift_stress,
                          inverse_spatial_metric, auxiliary_velocity);
}

// Currently these are the same because we always do the work of constructing
// the mixed version of the rotational-shift stress tensor

void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const Scalar<DataVector>& velocity_potential) {
  fluxes_on_face(flux_on_face, face_normal_vector, face_normal,
                 rotational_shift_stress, velocity_potential);
}

void Sources<Geometry::FlatCartesian>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_flat_potential_sources(equation_for_potential,
                             log_deriv_of_lapse_over_specific_enthalpy,
                             flux_for_potential);
}

void Sources<Geometry::Curved>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_curved_potential_sources(equation_for_potential,
                               log_deriv_of_lapse_over_specific_enthalpy,
                               christoffel_contracted, flux_for_potential);
}


}  // namespace IrrotationalBns
