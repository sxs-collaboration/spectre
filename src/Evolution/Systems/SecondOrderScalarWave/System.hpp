// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/TimeDerivative.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the second-order scalar wave equations.
 *
 * \f{align*}
 * \partial_t \Psi &= -\Pi \\
 * \partial_t \Pi &= -\delta^{ij}\partial_i \partial_j \Psi
 * \f}
 */
namespace SecondOrderScalarWave {

template <size_t Dim>
struct System {
  static std::string name() { return "SecondOrderScalarWave"; }

  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = ::Tags::Variables<tmpl::list<Tags::Psi, Tags::Pi>>;
  using flux_variables = tmpl::list<>;
  using auxiliary_variables = tmpl::list<Tags::Phi<Dim>>;
  // The time derivative reads only the derivative of Phi, but the evolved
  // variables must also be listed: `volume_terms` unconditionally instantiates
  // reads of every evolved variable's derivative for the moving-mesh term, so
  // a nonconservative system's `gradient_variables` must contain its evolved
  // variables.
  using gradient_variables = tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>;

  using compute_volume_time_derivative_terms = TimeDerivative<Dim>;
};
}  // namespace SecondOrderScalarWave
