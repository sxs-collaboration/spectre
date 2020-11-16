// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace NewtonianEuler {
/// \cond
namespace Sources {
struct NoSource;
}  // namespace Sources
/// \endcond

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
    const Scalar<DataVector>& pressure) noexcept;
}  // namespace detail

/*!
 * \brief Compute the time derivative of the conserved variables for the
 * Newtonian Euler system
 */
template <size_t Dim, typename InitialDataType>
struct TimeDerivativeTerms {
 private:
  struct EnthalpyDensity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using SourceTerm = typename InitialDataType::source_term_type;
  using argument_tags_flux =
      tmpl::list<Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
                 Tags::Velocity<DataVector, Dim>, Tags::Pressure<DataVector>>;

 public:
  using temporary_tags = tmpl::list<EnthalpyDensity>;
  using argument_tags = tmpl::append<
      argument_tags_flux,
      tmpl::conditional_t<std::is_same_v<SourceTerm, Sources::NoSource>,
                          tmpl::list<>,
                          tmpl::push_front<typename SourceTerm::argument_tags,
                                           Tags::SourceTerm<InitialDataType>>>>;

  template <typename... SourceTermArgs>
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
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure, const SourceTerm& source,
      const SourceTermArgs&... source_term_args) noexcept {
    detail::fluxes_impl(mass_density_cons_flux, momentum_density_flux,
                        energy_density_flux, enthalpy_density, momentum_density,
                        energy_density, velocity, pressure);
    if constexpr (not std::is_same_v<SourceTerm, Sources::NoSource>) {
      sources_impl(
          std::make_tuple(non_flux_terms_dt_mass_density,
                          non_flux_terms_dt_momentum_density,
                          non_flux_terms_dt_energy_density),
          typename InitialDataType::source_term_type::sourced_variables{},
          source, source_term_args...);
    }
  }

  static void apply(
      // Time derivatives returned by reference. No source terms or
      // nonconservative products, so not used. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<
          Scalar<DataVector>*> /*non_flux_terms_dt_mass_density*/,
      const gsl::not_null<
          tnsr::I<DataVector, Dim>*> /*non_flux_terms_dt_momentum_density*/,
      const gsl::not_null<
          Scalar<DataVector>*> /*non_flux_terms_dt_energy_density*/,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
      const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> enthalpy_density,

      // Arguments listed in argument_tags above
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure) noexcept {
    detail::fluxes_impl(mass_density_cons_flux, momentum_density_flux,
                        energy_density_flux, enthalpy_density, momentum_density,
                        energy_density, velocity, pressure);
  }

 private:
  using non_flux_terms_dt_vars_list =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>;

  template <typename... SourceTermArgs, typename... SourcedVars>
  static void sources_impl(std::tuple<gsl::not_null<Scalar<DataVector>*>,
                                      gsl::not_null<tnsr::I<DataVector, Dim>*>,
                                      gsl::not_null<Scalar<DataVector>*>>
                               non_flux_terms_dt_vars,
                           tmpl::list<SourcedVars...> /*meta*/,
                           const SourceTerm& source,
                           const SourceTermArgs&... source_term_args) noexcept {
    source.apply(
        std::get<
            tmpl::index_of<non_flux_terms_dt_vars_list, SourcedVars>::value>(
            non_flux_terms_dt_vars)...,
        source_term_args...);
  }
};

}  // namespace NewtonianEuler
