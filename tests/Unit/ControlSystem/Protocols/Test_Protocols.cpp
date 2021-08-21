// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "Utilities/ProtocolHelpers.hpp"

static_assert(
    tt::assert_conforms_to<control_system::TestHelpers::ExampleSubmeasurement,
                           control_system::protocols::Submeasurement>);
static_assert(
    tt::assert_conforms_to<control_system::TestHelpers::ExampleMeasurement,
                           control_system::protocols::Measurement>);
static_assert(
    tt::assert_conforms_to<control_system::TestHelpers::ExampleControlSystem,
                           control_system::protocols::ControlSystem>);
