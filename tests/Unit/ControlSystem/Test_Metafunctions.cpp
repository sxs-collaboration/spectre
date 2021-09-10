// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Metafunctions.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::metafunctions {
namespace {
struct LabelA;
struct LabelB;
struct LabelC;
struct Metavariables;

namespace Measurements {
using MeasurementA = control_system::TestHelpers::Measurement<LabelA>;
using MeasurementB = control_system::TestHelpers::Measurement<LabelB>;
using SystemA0 = control_system::TestHelpers::System<2, LabelA, MeasurementA>;
using SystemA1 = control_system::TestHelpers::System<2, LabelB, MeasurementA>;
using SystemB0 = control_system::TestHelpers::System<2, LabelC, MeasurementB>;
using ComponentA = ControlComponent<Metavariables, SystemA0>;
using ComponentB = ControlComponent<Metavariables, SystemB0>;

using test_systems = tmpl::list<SystemA0, SystemB0, SystemA1>;
using fewer_test_systems = tmpl::list<SystemA0, SystemB0>;

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

static_assert(
    std::is_same_v<control_components<Metavariables, fewer_test_systems>,
                   tmpl::list<ControlComponent<Metavariables, SystemA0>,
                              ControlComponent<Metavariables, SystemB0>>> or
    std::is_same_v<control_components<Metavariables, fewer_test_systems>,
                   tmpl::list<ControlComponent<Metavariables, SystemB0>,
                              ControlComponent<Metavariables, SystemA0>>>);
}  // namespace Measurements

namespace InterpolationTargetTags {
template <typename Systems>
struct Target;

struct SubmeasurementTarget
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  template <typename ControlSystems>
  using interpolation_target_tag = Target<ControlSystems>;

  using argument_tags = tmpl::list<>;
};

struct SubmeasurementVoid
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  template <typename ControlSystems>
  using interpolation_target_tag = void;

  using argument_tags = tmpl::list<>;
};

struct Measurement : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<SubmeasurementTarget, SubmeasurementVoid>;
};

struct MeasurementEmpty
    : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<>;
};

template <typename Label>
struct ControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "ControlSystem"; }
  using measurement = Measurement;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

struct ControlSystemEmpty
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "Empty"; }
  using measurement = MeasurementEmpty;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

static_assert(std::is_same_v<submeasurements<Measurement>::type,
                             Measurement::submeasurements>);
static_assert(std::is_same_v<submeasurements_t<Measurement>,
                             submeasurements<Measurement>::type>);

static_assert(
    std::is_same_v<
        interpolation_target_tags<tmpl::list<
            ControlSystem<LabelA>, ControlSystem<LabelB>, ControlSystemEmpty>>,
        tmpl::list<
            Target<tmpl::list<ControlSystem<LabelA>, ControlSystem<LabelB>>>>>);
}  // namespace InterpolationTargetTags
}  // namespace
}  // namespace control_system::metafunctions
