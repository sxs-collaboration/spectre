// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Phase.hpp"
#include "Parallel/PhaseControl/CheckpointAndExitAfterWallclock.hpp"
#include "Parallel/PhaseControl/VisitAndReturn.hpp"
#include "Utilities/TMPL.hpp"

namespace PhaseControl {
using factory_creatable_classes =
    tmpl::list<VisitAndReturn<Parallel::Phase::LoadBalancing>,
               VisitAndReturn<Parallel::Phase::WriteCheckpoint>,
               CheckpointAndExitAfterWallclock>;
}
