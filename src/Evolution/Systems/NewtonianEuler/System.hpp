// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/TimeDerivative.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the Newtonian Euler system
namespace NewtonianEuler {

template <size_t Dim, typename EquationOfStateType, typename InitialDataType>
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;
  static constexpr size_t thermodynamic_dim =
      EquationOfStateType::thermodynamic_dim;

  using variables_tag = ::Tags::Variables<tmpl::list<
      Tags::MassDensityCons, Tags::MomentumDensity<Dim>, Tags::EnergyDensity>>;
  using flux_variables =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables =
      typename InitialDataType::source_term_type::sourced_variables;
  // Compute item for pressure not currently implemented in SpECTRE,
  // so its simple tag is passed along with the primitive variables.
  using primitive_variables_tag = ::Tags::Variables<tmpl::list<
      Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, Dim>,
      Tags::SpecificInternalEnergy<DataVector>, Tags::Pressure<DataVector>>>;

  using compute_volume_time_derivative_terms =
      TimeDerivative<Dim, InitialDataType>;
  using volume_fluxes = ComputeFluxes<Dim>;
  using volume_sources = ComputeSources<InitialDataType>;


  using conservative_from_primitive = ConservativeFromPrimitive<Dim>;
  using primitive_from_conservative =
      PrimitiveFromConservative<Dim, thermodynamic_dim>;

  using char_speeds_compute_tag = Tags::CharacteristicSpeedsCompute<Dim>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed<Dim>;

  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

}  // namespace NewtonianEuler
