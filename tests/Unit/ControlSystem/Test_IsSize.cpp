// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ControlSystem/IsSize.hpp"
#include "ControlSystem/Measurements/BNSCenterOfMass.hpp"
#include "ControlSystem/Systems/Expansion.hpp"
#include "ControlSystem/Systems/Size.hpp"
#include "Domain/Structure/ObjectLabel.hpp"

namespace control_system {
namespace {
using yes_size = control_system::Systems::Size<domain::ObjectLabel::A, 3>;
using both_horizons = control_system::measurements::BothHorizons;
using no_size = control_system::Systems::Expansion<2, both_horizons>;

static_assert(size::is_size_v<yes_size>);
static_assert(not size::is_size_v<no_size>);
}  // namespace
}  // namespace control_system
