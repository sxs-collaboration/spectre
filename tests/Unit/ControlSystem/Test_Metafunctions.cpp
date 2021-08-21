// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "ControlSystem/Metafunctions.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::metafunctions {
namespace {
struct LabelA;
struct LabelB;
struct LabelC;

using MeasurementA = control_system::TestHelpers::Measurement<LabelA>;
using MeasurementB = control_system::TestHelpers::Measurement<LabelB>;
using SystemA0 = control_system::TestHelpers::System<LabelA, MeasurementA>;
using SystemA1 = control_system::TestHelpers::System<LabelB, MeasurementA>;
using SystemB0 = control_system::TestHelpers::System<LabelC, MeasurementB>;

using test_systems = tmpl::list<SystemA0, SystemB0, SystemA1>;

static_assert(std::is_same_v<measurement<SystemA0>::type, MeasurementA>);

static_assert(std::is_same_v<measurements_t<test_systems>,
                             measurements<test_systems>::type>);

// Order is unspecified
static_assert(std::is_same_v<measurements_t<test_systems>,
                             tmpl::list<MeasurementA, MeasurementB>> or
              std::is_same_v<measurements_t<test_systems>,
                             tmpl::list<MeasurementB, MeasurementA>>);

static_assert(
    std::is_same_v<
        control_systems_with_measurement_t<test_systems, MeasurementA>,
        control_systems_with_measurement<test_systems, MeasurementA>::type>);
static_assert(
    std::is_same_v<
        control_systems_with_measurement_t<test_systems, MeasurementB>,
        control_systems_with_measurement<test_systems, MeasurementB>::type>);

// Order is unspecified
static_assert(
    std::is_same_v<
        control_systems_with_measurement_t<test_systems, MeasurementA>,
        tmpl::list<SystemA0, SystemA1>> or
    std::is_same_v<
        control_systems_with_measurement_t<test_systems, MeasurementA>,
        tmpl::list<SystemA1, SystemA0>>);
static_assert(std::is_same_v<
              control_systems_with_measurement_t<test_systems, MeasurementB>,
              tmpl::list<SystemB0>>);
}  // namespace
}  // namespace control_system::metafunctions
