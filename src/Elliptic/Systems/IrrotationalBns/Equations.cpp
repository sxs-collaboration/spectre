// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Poisson/Equations.hpp"

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace IrrotationalBns {

void flat_potential_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  for (size_t d = 0; d < 3; d++) {
    flux_for_field->get(d) = auxiliary_velocity.get(d);
  }
}

void curved_fluxes(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  raise_or_lower_first_index(flux_for_potential, auxiliary_velocity,
                             inverse_spatial_metric);
}

void add_flat_cartesian_sources(
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_for_potential,
    const tnsr::I<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3> log_deriv_of_lapse_over_specific_enthalpy) {
  get(source_for_potential) +=
      auxiliary_velocity->get(0) *
      log_deriv_of_lapse_over_specific_enthalpy->get(0);
  get(source_for_potential) +=
      auxiliary_velocity->get(1) *
      log_deriv_of_lapse_over_specific_enthalpy->get(1);
  get(source_for_potential) +=
      auxiliary_velocity->get(2) *
      log_deriv_of_lapse_over_specific_enthalpy->get(2);
}

void add_curved_sources(const gsl::not_null<Scalar<DataVector>*>
                            source_for_potential const tnsr::i<DataVector, 3>
                                log_deriv_of_lapse_over_specific_enthalpy,
                        const tnsr::i<DataVector, 3>& christoffel_contracted,
                        const tnsr::I<DataVector, 3>& flux_for_potential) {
  // The flux for the potential is the auxiliary velocity
  // (when the index is in the right place)
  ::tenex::update(source_for_potential,
                  source_for_potential() +
                      flux_for_potential(ti::I) *
                          log_deriv_of_lapse_over_specific_enthalpy(ti::i));
  get(source_for_potential) -=
      get(dot_product(christoffel_contracted, flux_for_potential));
}

void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress);
{
  get(flux_for_auxiliary) = get(rotational_shift_stress);
  for (size_t d = 0; d < 3; d++) {
    flux_for_auxiliary->get(d, d) -= 1.0;
  }
  get(flux_for_auxiliary) *= get(velocity_potential);
}

void add_auxiliary_sources_without_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources) {
  get(auxiliary_sources) +=
      get(auxiliary_velocity) -
      get(velocity_potential) * get(div_rotational_shift_stress);
  get(auxiliary_sources) -= get(fixed_sources)
}
void add_auxiliary_source_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const tnsr::Ij<DataVector, 3>& flux_for_auxiliary,
    const tnsr::i<DataVector, 3>& christoffel_contracted) {
  get(source_for_potential) -=
      get(dot_product(christoffel_contracted, flux_for_potential));
}
void Fluxes<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  flat_cartesian_fluxes(flux_for_potential, auxiliary_velocity);
}

void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& auxiliary_velocity) {
  curved_fluxes(flux_for_potential, inv_spatial_metric, auxiliary_velocity);
}

void Fluxes<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const tnsr::II<DataVector, 3>& /*inv_spatial_metric*/,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential) {
  auxiliary_fluxes(flux_for_auxiailiary, velocity_potential,
                   rotational_shift_stress);
}
// Currently these are the same
void Fluxes<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const tnsr::II<DataVector, 3>& /*inv_spatial_metric*/,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential, ) {
  auxiliary_fluxes(flux_for_auxiailiary, velocity_potential,
                   rotational_shift_stress);
}

void Sources<Geometry::FlatCartesian>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& /*auxiliary_velocity*/,
    const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& /*fixed_sources*/,
    const tnsr::i<DataVector, 3>& /*potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_flat_cartesian_sources(equation_for_potential, flux_for_potential,
                             log_deriv_of_lapse_over_specific_enthalpy);
}

void Sources<Geometry::Curved>::apply(
    const gsl::not_null<Scalar<DataVector>*> equation_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
    const tnsr::i<DataVector, 3>& /*auxiliary_velocity*/,
    const tnsr::i<DataVector, 3>& /*fixed_sources*/,
    const tnsr::Ij<DataVector, 3>& /*flux_for_auxiliary*/,
    const tnsr::i<DataVector, 3>& /*velocity_potential*/,
    const tnsr::I<DataVector, 3>& flux_for_potential) {
  add_curved_sources(equation_for_field,
                     log_deriv_of_lapse_over_specific_enthalpy,
                     christoffel_contracted, flux_for_potential);
}

void Sources<Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        equation_for_auxiliary_velocity,
    const tnsr::i<DataVector, 3>& /*log_deriv_of_lapse_over_specific_enthalpy*/
    const tnsr::i<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources,
    const Scalar<DataVector>& velocity_potential) {
  add_auxiliary_sources_without_flux_christoffels(
      equation_for_auxiliary_velocity, velocity_potential, auxiliary_velocity,
      div_rotational_shift_stress, fixed_sources, );
}

void Sources<Geometry::Curved>::apply(
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        equation_for_auxiliary_velocity,
    const tnsr::i<DataVector, 3>& /*log_deriv_of_lapse_over_specific_enthalpy*/,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3>& fixed_sources,
    const tnsr::Ij<DataVector, 3>& flux_for_auxiliary,
    const Scalar<DataVector>& velocity_potential) {
  add_auxiliary_sources_without_flux_christoffels(
      equation_for_auxiliary_velocity, velocity_potential, auxiliary_velocity,
      div_rotational_shift_stress, fixed_sources);
  add_auxiliary_source_flux_christoffels(equation_for_auxiliary_velocity,
                                         flux_for_auxiliary,
                                         christoffel_contracted);
}

}  // namespace IrrotationalBns
