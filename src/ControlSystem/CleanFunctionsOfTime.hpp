// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "IO/Observer/ObserverComponent.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace control_system::Tags {
struct MeasurementTimescales;
}  // namespace control_system::Tags
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace control_system {
/// \ingroup ControlSystemGroup
/// Reduction action that deletes function-of-time data from before
/// the passed time.
///
/// \see CleanFunctionsOfTime
struct CleanFunctionsOfTimeAction {
 private:
  struct CleanFunc {
    static void apply(
        gsl::not_null<std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
            functions,
        double time);
  };

 public:
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/, const double time) {
    Parallel::mutate<domain::Tags::FunctionsOfTime, CleanFunc>(cache, time);
    Parallel::mutate<Tags::MeasurementTimescales, CleanFunc>(cache, time);
  }
};

/// \ingroup ControlSystemGroup
/// Delete old function-of-time data to save memory.
///
/// \warning Running this event will make the
/// `global_functions_of_time` data stored in the output files
/// useless.
class CleanFunctionsOfTime : public Event {
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>>;

 public:
  /// \cond
  WRAPPED_PUPable_decl_template(CleanFunctionsOfTime);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Delete old function-of-time data to save memory.\n"
      "\n"
      "WARNING: Running this event will make the global_functions_of_time\n"
      "data stored in the output files useless.";

  CleanFunctionsOfTime() = default;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::Time>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(const double time,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    if constexpr (Parallel::is_dg_element_collection_v<ParallelComponent>) {
      (void)cache;
      (void)array_index;
      (void)time;
      ERROR("Reductions not implemented with DgElementCollection.");
    } else {
      // It doesn't matter what individual component the target is.
      // We use ObserverWriter[0] because it must exist and is easy to
      // name here.
      const auto& writer0_proxy = Parallel::get_parallel_component<
          observers::ObserverWriter<Metavariables>>(cache)[0];
      const auto& self_proxy =
          Parallel::get_parallel_component<ParallelComponent>(
              cache)[array_index];
      Parallel::contribute_to_reduction<CleanFunctionsOfTimeAction>(
          ReductionData(time), self_proxy, writer0_proxy);
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override;
};
}  // namespace control_system
