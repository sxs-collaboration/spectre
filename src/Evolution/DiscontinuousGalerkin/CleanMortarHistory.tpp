// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
template <typename System>
void CleanMortarHistory<System>::apply(
    const gsl::not_null<DirectionalIdMap<
        dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<dim>,
                                           ::evolution::dg::MortarData<dim>,
                                           CouplingResult>>*>
        history,
    const LtsTimeStepper& time_stepper,
    const DirectionalIdMap<dim, MortarInfo<dim>>& mortar_info) {
  for (auto& [mortar_id, hist] : *history) {
    const auto time_stepping_policy =
        mortar_info.at(mortar_id).time_stepping_policy();
    switch (time_stepping_policy) {
      case TimeSteppingPolicy::EqualRate:
        break;
      case TimeSteppingPolicy::Conservative:
        time_stepper.clean_boundary_history(make_not_null(&hist));
        break;
      default:
        ERROR("Unhandled TimeSteppingPolicy: " << time_stepping_policy);
    }
  }
}
}  // namespace evolution::dg
