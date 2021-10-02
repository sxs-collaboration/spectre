// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/Protocols/ControlSystem.hpp"
#include "ControlSystem/Protocols/Measurement.hpp"
#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "ControlSystem/RunCallbacks.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/Actions/UpdateMessageQueue.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system::TestHelpers {
template <typename ControlSystem>
struct ReplaceThisWithControlSystemComponentWhenThatIsWritten;

struct SomeTagOnElement : db::SimpleTag {
  using type = double;
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
        box, [&time, &data](const gsl::not_null<double*> stored_time,
                            const gsl::not_null<double*> stored_data) {
          *stored_time = time;
          *stored_data = tuples::get<SubmeasurementTag>(data);
        });
  }
};

/// [Submeasurement]
struct ExampleSubmeasurement
    : tt::ConformsTo<control_system::protocols::Submeasurement> {
  // This submeasurement does not use an interpolation component.
  template <typename ControlSystems>
  using interpolation_target_tag = void;

  using argument_tags = tmpl::list<SomeTagOnElement>;

  template <typename Metavariables, typename ParallelComponent,
            typename ControlSystems>
  static void apply(const double data_from_element,
                    const LinkedMessageId<double>& measurement_id,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const int& /*array_index*/,
                    const ParallelComponent* /*meta*/,
                    ControlSystems /*meta*/) {
    // In real cases, we would generally do a reduction to a single
    // chare here and have the below code in the reduction action, but
    // the action testing framework doesn't support reductions, so
    // this example is for a control system running on a singleton and
    // we just pass the data through.
    const auto box =
        db::create<db::AddSimpleTags<MeasurementResultTag>>(data_from_element);
    control_system::RunCallbacks<ExampleSubmeasurement, ControlSystems>::apply(
        box, cache, measurement_id);
  }
};
/// [Submeasurement]

/// [Measurement]
struct ExampleMeasurement
    : tt::ConformsTo<control_system::protocols::Measurement> {
  using submeasurements = tmpl::list<ExampleSubmeasurement>;
};
/// [Measurement]

/// [ControlSystem]
struct ExampleControlSystem
    : tt::ConformsTo<control_system::protocols::ControlSystem> {
  static std::string name() { return "ExampleControlSystem"; }
  using measurement = ExampleMeasurement;

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
          ReplaceThisWithControlSystemComponentWhenThatIsWritten<
              ExampleControlSystem>>(cache);
      Parallel::simple_action<Actions::UpdateMessageQueue<
          ExampleSubmeasurementQueueTag, MeasurementQueue,
          SomeControlSystemUpdater>>(control_system_proxy, measurement_id,
                                     measurement_result);
    }
  };
};
/// [ControlSystem]
}  // namespace control_system::TestHelpers
