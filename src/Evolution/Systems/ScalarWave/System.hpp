// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/TimeDerivative.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar wave equation.
 *
 * The equations of motion for the system augmented with constraint damping
 * terms are given by Eq. (15), (23) and (24) of \cite Holst2004wt (setting
 * background spacetime to Minkowskian):
 *
 * \f{align*}
 * \partial_t \psi =& -\Pi \\
 * \partial_t \Pi  =& -\partial^i \Phi_i \\
 * \partial_t \Phi_i =& -\partial_i \Pi + \gamma_2 (\partial_i \psi - \Phi_i)
 * \f}
 *
 * In our implementation here, to disable the constraint damping terms,
 * set \f$\gamma_2 = 0\f$.
 */
namespace ScalarWave {

template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = ::Tags::Variables<tmpl::list<Pi, Phi<Dim>, Psi>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<Pi, Phi<Dim>, Psi>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;

  using compute_largest_characteristic_speed =
      Tags::ComputeLargestCharacteristicSpeed;

  using char_speeds_compute_tag = Tags::CharacteristicSpeedsCompute<Dim>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim>;

  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;

  // Remove gradients_tags once GH is converted over to the new
  // dg::ComputeTimeDerivative action.
  using gradients_tags = gradient_variables;
};
}  // namespace ScalarWave
