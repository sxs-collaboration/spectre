// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"   // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {
namespace Events {
template <typename ObservationValueTag, typename EventRegistrars>
class ObserveTime;

namespace Registrars {
template <typename ObservationValueTag>
using ObserveTime =
    ::Registration::Registrar<Events::ObserveTime, ObservationValueTag>;
}  // namespace Registrars

template <typename ObservationValueTag,
          typename EventRegistrars =
              tmpl::list<Registrars::ObserveTime<ObservationValueTag>>>
class ObserveTime;  // IWYU pragma: keep

/*!
 * Add docs
 */
template <typename ObservationValueTag, typename EventRegistrars>
class ObserveTime : public Event<EventRegistrars> {
 private:
  using ReductionData = Parallel::ReductionData<
      Parallel::ReductionDatum<std::string, funcl::AssertEqual<>>>;

 public:
  /// \cond
  explicit ObserveTime(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTime);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help = "Help string needed";

  ObserveTime() = default;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags = tmpl::list<ObservationValueTag>;

  template <typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  void operator()(
      const db::const_item_type<ObservationValueTag>& observation_value,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) const noexcept {
    // Send data to reduction observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();

    std::string info_to_print =
        "Current time: " +
        std::to_string(static_cast<double>(observation_value));

    Parallel::simple_action<observers::Actions::ContributeReductionData<
        observers::ThreadedActions::PrintReductionData>>(
        local_observer,
        observers::ObservationId(
            observation_value,
            typename Metavariables::element_observation_type{}),
        ReductionData{info_to_print});
  }
};

/// \cond
template <typename ObservationValueTag, typename EventRegistrars>
PUP::able::PUP_ID ObserveTime<ObservationValueTag, EventRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
