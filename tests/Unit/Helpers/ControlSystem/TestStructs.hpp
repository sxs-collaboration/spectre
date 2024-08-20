// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Helpers/ControlSystem/Examples.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Tags {
struct PreviousTriggerTime;
}  // namespace Tags

namespace control_system::TestHelpers {
namespace TestStructs_detail {
struct LabelA {};
}  // namespace TestStructs_detail

template <typename Label, typename ControlSystems, bool CallRunCallbacks>
class TestEvent;

template <typename Label, bool CallRunCallbacks>
struct Submeasurement
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  template <typename ControlSystems>
  using interpolation_target_tag = void;
  template <typename ControlSystems>
  using event = TestEvent<Label, ControlSystems, CallRunCallbacks>;
};

template <typename Label, typename ControlSystems, bool CallRunCallbacks>
class TestEvent : public ::Event {
 public:
  /// \cond
  // LCOV_EXCL_START
  explicit TestEvent(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestEvent);  // NOLINT
  // LCOV_EXCL_STOP
  /// \endcond

  static constexpr bool factory_creatable = false;
  TestEvent() = default;

  using compute_tags_for_observation_box = tmpl::list<>;
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::conditional_t<
      CallRunCallbacks,
      tmpl::list<::Tags::Time, ::Tags::PreviousTriggerTime,
                 control_system::TestHelpers::SomeTagOnElement>,
      tmpl::list<>>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const double time, const std::optional<double> previous_time,
                  const double data_from_element,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    const LinkedMessageId<double> measurement_id{time, previous_time};
    // Just a hack so we don't have to add a whole other component
    const auto box = db::create<
        db::AddSimpleTags<control_system::TestHelpers::MeasurementResultTag>>(
        data_from_element);
    control_system::RunCallbacks<Submeasurement<Label, CallRunCallbacks>,
                                 ControlSystems>::apply(box, cache,
                                                        measurement_id);
    ++call_count;
  }

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    ++call_count;
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*component*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  static size_t call_count; // NOLINT
};

// NOLINTBEGIN
template <typename Label, typename ControlSystems, bool CallRunCallbacks>
size_t TestEvent<Label, ControlSystems, CallRunCallbacks>::call_count = 0;

/// \cond
template <typename Label, typename ControlSystems, bool CallRunCallbacks>
PUP::able::PUP_ID
    TestEvent<Label, ControlSystems, CallRunCallbacks>::my_PUP_ID =
        0;
/// \endcond
// NOLINTEND

template <typename Label, bool CallRunCallbacks = false>
struct Measurement : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<Submeasurement<Label, CallRunCallbacks>>;
};

struct ControlError : tt::ConformsTo<control_system::protocols::ControlError> {
  using object_centers = domain::object_list<>;
  void pup(PUP::er& /*p*/) {}

  using options = tmpl::list<>;
  static constexpr Options::String help{"Example control error."};

  template <typename Metavariables, typename... QueueTags>
  DataVector operator()(
      const ::TimescaleTuner<true>& /*tuner*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const double /*time*/, const std::string& /*function_of_time_name*/,
      const tuples::TaggedTuple<QueueTags...>& /*measurements*/) {
    return DataVector{};
  }
};

static_assert(tt::assert_conforms_to_v<Measurement<TestStructs_detail::LabelA>,
                                       control_system::protocols::Measurement>);

template <size_t DerivOrder, typename Label, typename Measurement>
struct System : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return pretty_type::short_name<Label>(); }
  static std::optional<std::string> component_name(
      const size_t i, const size_t /*num_components*/) {
    return get_output(i);
  }
  using measurement = Measurement;
  using simple_tags = tmpl::list<>;
  using control_error = ControlError;
  static constexpr size_t deriv_order = DerivOrder;

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<>;
  };
};

static_assert(
    tt::assert_conforms_to_v<System<2, TestStructs_detail::LabelA,
                                    Measurement<TestStructs_detail::LabelA>>,
                             control_system::protocols::ControlSystem>);
}  // namespace control_system::TestHelpers
