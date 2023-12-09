// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/IrrotationalBns/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace IrrotationalBns {

void flat_potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  for (size_t d = 0; d < 3; d++) {
    flux_for_potential->get(d) = auxiliary_velocity.get(d);
  }
}

void curved_potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  ::tenex::evaluate<ti::I>(
      flux_for_potential,
      inverse_spatial_metric(ti::I, ti::J) * auxiliary_velocity(ti::j));
}

void add_flat_potential_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::I<DataVector, 3>& upper_auxiliary_velocity,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy) {
  for (size_t i = 0; i < 3; i++) {
    source_for_potential->get() -=
        upper_auxiliary_velocity.get(i) *
        log_deriv_of_lapse_over_specific_enthalpy.get(i);
  }
}

void add_curved_potential_sources(
    const gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  // The flux for the potential is the auxiliary velocity
  // (when the index is in the right place)
  ::tenex::update(
      source_for_potential,
      (*source_for_potential)() +
          flux_for_potential(ti::I) *
              log_deriv_of_lapse_over_specific_enthalpy(ti::i) -
          christoffel_contracted(ti::i) * flux_for_potential(ti::I));
}

void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress) {
  tenex::evaluate<ti::I, ti::j>(flux_for_auxiliary,
                                -rotational_shift_stress(ti::I, ti::j) / 2.0);
  for (size_t d = 0; d < 3; d++) {
    flux_for_auxiliary->get(d, d) += 1.0;
  }
  ::tenex::update<ti::I, ti::j>(
      flux_for_auxiliary,
      (*flux_for_auxiliary)(ti::I, ti::j) * velocity_potential());
}

void add_auxiliary_sources_without_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources) {
  ::tenex::update<ti::i>(auxiliary_sources,
                         (*auxiliary_sources)(ti::i)-velocity_potential() *
                                 div_rotational_shift_stress(ti::i) / 2 -
                             fixed_sources(ti::i));
}
void add_auxiliary_source_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::Ijj<DataVector, 3>& christoffel,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress) {
  ::tenex::update<ti::i>(auxiliary_sources,
                         (*auxiliary_sources)(ti::i)-velocity_potential() *
                             (christoffel(ti::K, ti::j, ti::i) *
                                  rotational_shift_stress(ti::J, ti::k) -
                              christoffel_contracted(ti::j) *
                                  rotational_shift_stress(ti::J, ti::i)));
}
// Flat-space cartesian fluxes

void Fluxes<Geometry::FlatCartesian>::apply(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  flat_potential_fluxes(flux_for_potential, auxiliary_velocity);
}

void Fluxes<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential) {
  auxiliary_fluxes(flux_for_auxiliary, velocity_potential,
                   rotational_shift_stress);
}

// Curved-space (generic fluxes)

void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  curved_potential_fluxes(flux_for_potential, inverse_spatial_metric,
                          auxiliary_velocity);
}

// Currently these are the same because we always do the work of constructing
// the mixed version of the rotational-shift stress tensor

void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const tnsr::II<DataVector, 3>& /*inv_spatial_metric*/,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential) {
  auxiliary_fluxes(flux_for_auxiliary, velocity_potential,
                   rotational_shift_stress);
}

void Sources<Geometry::FlatCartesian>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& /*fixed_sources*/,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_flat_potential_sources(equation_for_potential, flux_for_potential,
                             log_deriv_of_lapse_over_specific_enthalpy);
}

// Need auxiliary velocity here (since don't have fluxes)
void Sources<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        equation_for_auxiliary_velocity,
    const tnsr::i<DataVector, 3>& /*log_deriv_of_lapse_over_specific_enthalpy*/,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources,
    const Scalar<DataVector>& velocity_potential) {
  add_auxiliary_sources_without_flux_christoffels(
      equation_for_auxiliary_velocity, velocity_potential,
      div_rotational_shift_stress, fixed_sources);
}

void Sources<Geometry::Curved>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::Ijj<DataVector, 3>& /*christoffel*/,
    const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
    const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& /*fixed_sources*/,
    const Scalar<DataVector>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_curved_potential_sources(equation_for_potential,
                               log_deriv_of_lapse_over_specific_enthalpy,
                               christoffel_contracted, flux_for_potential);
}

// Need auxiliary velocity here (again because we don't have fluxes)
void Sources<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        equation_for_auxiliary_velocity,
    const tnsr::i<DataVector, 3>& /*log_deriv_of_lapse_over_specific_enthalpy*/,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::Ijj<DataVector, 3>& christoffel,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources,
    const Scalar<DataVector>& velocity_potential) {
  add_auxiliary_sources_without_flux_christoffels(
      equation_for_auxiliary_velocity, velocity_potential,
      div_rotational_shift_stress, fixed_sources);
  add_auxiliary_source_flux_christoffels(
      equation_for_auxiliary_velocity, velocity_potential,
      christoffel_contracted, christoffel, rotational_shift_stress);
}

}  // namespace IrrotationalBns
