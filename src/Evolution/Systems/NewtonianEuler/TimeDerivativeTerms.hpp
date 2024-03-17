// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
class DataVector;
/// \endcond

namespace NewtonianEuler {
namespace detail {
template <size_t Dim>
void fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,
    gsl::not_null<Scalar<DataVector>*> enthalpy_density,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure);
}  // namespace detail

/*!
 * \brief Compute the time derivative of the conserved variables for the
 * Newtonian Euler system
 */
template <size_t Dim>
struct TimeDerivativeTerms {
 private:
  struct EnthalpyDensity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using temporary_tags = tmpl::list<EnthalpyDensity>;
  using argument_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::EquationOfState<false, 2>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time,
                 NewtonianEuler::Tags::SourceTerm<Dim>>;

  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim>*>
          non_flux_terms_dt_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
      const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> enthalpy_density,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, Dim>& coords, const double time,
      const Sources::Source<Dim>& source) {
    detail::fluxes_impl(mass_density_cons_flux, momentum_density_flux,
                        energy_density_flux, enthalpy_density, momentum_density,
                        energy_density, velocity, pressure);

    get(*non_flux_terms_dt_mass_density) = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      non_flux_terms_dt_momentum_density->get(i) = 0.0;
    }
    get(*non_flux_terms_dt_energy_density) = 0.0;
    const auto eos_2d = eos.promote_to_2d_eos();
    source(non_flux_terms_dt_mass_density, non_flux_terms_dt_momentum_density,
           non_flux_terms_dt_energy_density, mass_density_cons,
           momentum_density, energy_density, velocity, pressure,
           specific_internal_energy, *eos_2d, coords, time);
  }
};

}  // namespace NewtonianEuler
