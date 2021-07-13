// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Protocols.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeTransitionFunctions.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace domain::CoordinateMaps {
namespace {

static_assert(
    tt::assert_conforms_to<SphereTransition, protocols::TransitionFunc>);

}  // namespace
}  // namespace domain::CoordinateMaps
