// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Fluxes.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Sources.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TimeDerivativeTerms.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the relativistic Euler system
namespace RelativisticEuler {
/// \brief The Valencia formulation of the relativistic Euler System
/// See Chapter 7 of Relativistic Hydrodynamics by Luciano Rezzolla and Olindo
/// Zanotti or http://iopscience.iop.org/article/10.1086/303604
namespace Valencia {

template <size_t Dim, typename EquationOfStateType>
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;
  static constexpr size_t thermodynamic_dim =
      EquationOfStateType::thermodynamic_dim;

  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<Dim>>>;
  using flux_variables =
      tmpl::list<Tags::TildeD, Tags::TildeTau, Tags::TildeS<Dim>>;
  using gradient_variables = tmpl::list<>;
  // Source for TildeD is zero
  using sourced_variables = tmpl::list<Tags::TildeTau, Tags::TildeS<Dim>>;
  using primitive_variables_tag = ::Tags::Variables<
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>>>;
  using spacetime_variables_tag =
      ::Tags::Variables<gr::tags_for_hydro<Dim, DataVector>>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms<Dim>;
  using volume_fluxes = ComputeFluxes<Dim>;
  using volume_sources = ComputeSources<Dim>;

  using conservative_from_primitive = ConservativeFromPrimitive<Dim>;
  using primitive_from_conservative =
      PrimitiveFromConservative<thermodynamic_dim, Dim>;

  using char_speeds_compute_tag = Tags::CharacteristicSpeedsCompute<Dim>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim>;

  template <typename Tag>
  using magnitude_tag = ::Tags::NonEuclideanMagnitude<
      Tag, gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>;
};

}  // namespace Valencia
}  // namespace RelativisticEuler
