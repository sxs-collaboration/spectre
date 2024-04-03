// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaRDriftOutward.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::size::States {
using factory_creatable_states =
    tmpl::list<AhSpeed, DeltaR, DeltaRDriftOutward, Initial>;
}  // namespace control_system::size::States
