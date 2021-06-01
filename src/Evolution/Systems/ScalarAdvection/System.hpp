// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/ScalarAdvection/Characteristics.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/TimeDerivativeTerms.hpp"
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

  using boundary_conditions_base = BoundaryConditions::BoundaryCondition<Dim>;
  using boundary_correction_base = BoundaryCorrections::BoundaryCorrection<Dim>;

  using variables_tag = ::Tags::Variables<tmpl::list<Tags::U>>;
  using flux_variables = tmpl::list<Tags::U>;
  using gradient_variables = tmpl::list<>;
  using sourced_variables = tmpl::list<>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms<Dim>;

  using volume_fluxes = Fluxes<Dim>;

  using compute_largest_characteristic_speed =
      Tags::LargestCharacteristicSpeedCompute<Dim>;
};
}  // namespace ScalarAdvection
