// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Conservative/ConservativeDuDt.hpp"
#include "Evolution/Systems/Burgers/Equations.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \ingroup EvolutionSystemsGroup
/// \brief Items related to evolving the %Burgers equation
/// \f$0 = \partial_t U + \partial_x\left(U^2/2\right)\f$.
///
/// \note For this definition (i.e., with the factor of one half in the flux)
/// of the Burgers system, the local characteristic speed is \f$U\f$.
namespace Burgers {
struct System {
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = 1;

  using variables_tag = ::Tags::Variables<tmpl::list<Tags::U>>;
  using sourced_variables = tmpl::list<>;

  using compute_time_derivative = ConservativeDuDt<System>;
  using volume_fluxes = Fluxes;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed;

  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};
}  // namespace Burgers
