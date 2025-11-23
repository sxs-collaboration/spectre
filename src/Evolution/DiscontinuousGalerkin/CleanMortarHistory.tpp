// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"

#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
template <typename System>
void CleanMortarHistory<System>::apply(
    const gsl::not_null<DirectionalIdMap<
        dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<dim>,
                                           ::evolution::dg::MortarData<dim>,
                                           CouplingResult>>*>
        history,
    const LtsTimeStepper& time_stepper) {
  for (auto& mortar : *history) {
    time_stepper.clean_boundary_history(make_not_null(&mortar.second));
  }
}
}  // namespace evolution::dg
