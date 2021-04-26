// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar advection equation.
 *
 * \f{align*}
 * \partial_t U + \nabla \cdot (v U) = 0
 * \f}
 *
 * Since the ScalarAdvection system is only used for testing limiters in the
 * current implementation, the velocity field \f$v\f$ is fixed throughout time.
 */
namespace ScalarAdvection {
template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = ::Tags::Variables<tmpl::list<Tags::U>>;
  using flux_variables = tmpl::list<Tags::U>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables = tmpl::list<>;

  using volume_fluxes = Fluxes<Dim>;
};
}  // namespace ScalarAdvection
