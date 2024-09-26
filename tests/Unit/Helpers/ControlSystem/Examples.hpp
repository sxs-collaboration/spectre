// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>

#include "ControlSystem/Component.hpp"
#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageId.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system::TestHelpers {
struct SomeTagOnElement : db::SimpleTag {
  using type = double;
};

struct SomeOtherTagOnElement : db::SimpleTag {
  using type = double;
};

struct SomeOtherTagOnElementCompute : SomeOtherTagOnElement, db::ComputeTag {
  using base = SomeOtherTagOnElement;
  using return_type = double;
  using argument_tags = tmpl::list<>;
  static void function(const gsl::not_null<double*> result) { *result = 0.0; }
};

struct MeasurementResultTag : db::SimpleTag {
  using type = double;
};

struct MeasurementResultTime : db::SimpleTag {
  using type = double;
};

struct SomeControlSystemUpdater {
  // The measurement tags will probably not be template parameters in
  // most cases, but it makes it easier to make a clean example where
  // this is called.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename SubmeasurementTag>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, const double time,
                    tuples::TaggedTuple<SubmeasurementTag> data) {
    db::mutate<MeasurementResultTime, MeasurementResultTag>(
        [&time, &data](const gsl::not_null<double*> stored_time,
                       const gsl::not_null<double*> stored_data) {
          *stored_time = time;
          *stored_data = tuples::get<SubmeasurementTag>(data);
        },
        box);
  }
};

template <typename ControlSystems>
struct SomeEvent {};

/// [Submeasurement]
struct ExampleSubmeasurement
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  // This submeasurement does not use an interpolation component.
  template <typename ControlSystems>
  using interpolation_target_tag = void;

  template <typename ControlSystems>
  using event = SomeEvent<ControlSystems>;
};
/// [Submeasurement]

/// [Measurement]
struct ExampleMeasurement
    : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<ExampleSubmeasurement>;
};
/// [Measurement]

/// [ControlError]
struct ExampleControlError
    : tt::ConformsTo<control_system::protocols::ControlError> {
  using object_centers = domain::object_list<domain::ObjectLabel::A>;

  void pup(PUP::er& /*p*/) {}

  template <typename Metavariables, typename... QueueTags>
  DataVector operator()(const ::TimescaleTuner<true>& tuner,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<QueueTags...>& measurements) {
    const auto& functions_of_time =
        Parallel::get<domain::Tags::FunctionsOfTime>(cache);
    const double current_map_value =
        functions_of_time.at(function_of_time_name)->func(time)[0][0];
    const double measured_value = 0.0;
    // Would do something like get<QueueTag>(measurements) here
    (void)measurements;
    // Size control needs the timescale tuner
    (void)tuner;

    return {current_map_value - measured_value};
  }
};
/// [ControlError]

/// [ControlSystem]
struct ExampleControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "ExampleControlSystem"; }
  static std::optional<std::string> component_name(
      const size_t i, const size_t num_components) {
    ASSERT(num_components == 3,
           "This control system expected 3 components but there are "
               << num_components << " instead.");
    return i == 0 ? "X" : (i == 1 ? "Y" : "Z");
  }
  using measurement = ExampleMeasurement;

  using simple_tags = tmpl::list<>;

  static constexpr size_t deriv_order = 2;

  using control_error = ExampleControlError;

  // This is not part of the required interface, but is used by this
  // control system to store the measurement data.  Most control
  // systems will do something like this.
  struct ExampleSubmeasurementQueueTag {
    using type = double;
  };

  // As with the previous struct, this is not part of the required
  // interface.
  struct MeasurementQueue : db::SimpleTag {
    using type =
        LinkedMessageQueue<double, tmpl::list<ExampleSubmeasurementQueueTag>>;
  };

  struct process_measurement {
    template <typename Submeasurement>
    using argument_tags = tmpl::list<MeasurementResultTag>;

    template <typename Metavariables>
    static void apply(ExampleSubmeasurement /*meta*/,
                      const double measurement_result,
                      Parallel::GlobalCache<Metavariables>& cache,
                      const LinkedMessageId<double>& measurement_id) {
      // Process the submeasurement results and send whatever is
      // necessary to the control system component.  Usually calls
      // some simple action.
      auto& control_system_proxy = Parallel::get_parallel_component<
          ControlComponent<Metavariables, ExampleControlSystem>>(cache);
      Parallel::simple_action<::Actions::UpdateMessageQueue<
          ExampleSubmeasurementQueueTag, MeasurementQueue,
          SomeControlSystemUpdater>>(control_system_proxy, measurement_id,
                                     measurement_result);
    }
  };
};
/// [ControlSystem]
}  // namespace control_system::TestHelpers
