// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Cce {
/// Contains utilities for collecting, interpolating, and providing worldtube
/// data for CCE that originates from other components in a running simulation.
namespace InterfaceManagers{

/*!
 * \brief Enumeration of possibilities for the collection of worldtube data that
 * will be collected by the interpolator.
 *
 * \details The available strategies are:
 * - `InterpolationStrategy::EveryStep` : interpolation data will be provided on
 *   each full step for the time-stepper, so the worldtube data represents
 *   physical values that can then be interpolated/extrapolated.
 * - `InterpolationStrategy::EverySubstep` : interpolation data will be provided
 *   on each substep of the time-stepper, so the worldtube data represents
 *   intermediate substep quantities, and typically must be used in the same
 *   time-stepper to obtain physically meaningful quantites. This strategy is
 *   primarily useful if the CCE system takes identical timesteps using an
 *   identical stepper as the step and stepper used for generating the
 *   interpolation data.
 */
enum class InterpolationStrategy { EveryStep, EverySubstep };

/*!
 * \brief Determines whether the element should interpolate for the current
 * state of its `box`, depending on the
 * `Cce::InterfaceManagers::InterpolationStrategy` used by the associated
 * `Cce::InterfaceManagers::GhInterfaceManager`.
 */
template <typename DbTagList>
bool should_interpolate_for_strategy(
    const db::DataBox<DbTagList>& box,
    const InterpolationStrategy strategy) noexcept {
  if(strategy == InterpolationStrategy::EverySubstep) {
    return true;
  }
  if(strategy == InterpolationStrategy::EveryStep) {
    return db::get<::Tags::TimeStepId>(box).substep() == 0;
  }
  ERROR("Interpolation strategy not recognized");
}
}  // namespace InterfaceManagers
}  // namespace Cce
