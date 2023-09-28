
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"

namespace hydro {

/// @{
/*!
 * \brief Wrapper to add temperature variable to initial data
 * providing only density and or energy_density initialization.
 */

template <typename DerivedSolution>
class TemperatureInitialization {
 private:
  template <typename DataType, size_t Dim, typename... Args>
  auto variables_impl(const tnsr::I<DataType, Dim>& x,
                      tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/,
                      Args&... extra_args) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    const auto* derived = static_cast<DerivedSolution const*>(this);
    const auto& eos = derived->equation_of_state();
    if constexpr (std::decay_t<decltype(eos)>::thermodynamic_dim == 1) {
      return eos.temperature_from_density(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
    } else if constexpr (std::decay_t<decltype(eos)>::thermodynamic_dim == 2) {
      return eos.temperature_from_density_and_energy(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
          get<hydro::Tags::SpecificInternalEnergy<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
    } else {
      return eos.temperature_from_density_and_energy(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
          get<hydro::Tags::SpecificInternalEnergy<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
          get<hydro::Tags::ElectronFraction<DataType>>(derived->variables(
              x, extra_args...,
              tmpl::list<hydro::Tags::ElectronFraction<DataType>>{})));
    }
  }

 public:
  template <typename DataType, size_t Dim>
  auto variables(const tnsr::I<DataType, Dim>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    return variables_impl(x, tmpl::list<hydro::Tags::Temperature<DataType>>{});
  }

  template <typename DataType, size_t Dim>
  auto variables(const tnsr::I<DataType, Dim>& x, const double t,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    return variables_impl(x, tmpl::list<hydro::Tags::Temperature<DataType>>{},
                          t);
  }

  template <typename ExtraVars, typename DataType, size_t Dim, typename... Args>
  auto variables(ExtraVars& extra_variables, const tnsr::I<DataType, Dim>& x,
                 Args&... extra_args,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    const auto* derived = static_cast<DerivedSolution const*>(this);
    const auto& eos = derived->equation_of_state();
    if constexpr (std::decay_t<decltype(eos)>::thermodynamic_dim == 1) {
      return eos.temperature_from_density(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})));
    } else if constexpr (std::decay_t<decltype(eos)>::thermodynamic_dim == 2) {
      return eos.temperature_from_density_and_energy(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
          get<hydro::Tags::SpecificInternalEnergy<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})));
    } else {
      return eos.temperature_from_density_and_energy(
          get<hydro::Tags::RestMassDensity<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
          get<hydro::Tags::SpecificInternalEnergy<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
          get<hydro::Tags::ElectronFraction<DataType>>(derived->variables(
              extra_variables, x, extra_args...,
              tmpl::list<hydro::Tags::ElectronFraction<DataType>>{})));
    }
  }
};
/// @}

}  // namespace hydro
